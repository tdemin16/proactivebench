import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaOnevisionForConditionalGeneration,
)

try:
    from proactivebench.data_utils import (
        apply_conversation_template,
        load_image,
        load_proactivebench_dataset,
    )
    from proactivebench.environment import get_environment
    from proactivebench.open_ended_gen import get_oeg_judge_messages, parse_judge_prediction
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from proactivebench.data_utils import (
        apply_conversation_template,
        load_image,
        load_proactivebench_dataset,
    )
    from proactivebench.environment import get_environment
    from proactivebench.open_ended_gen import get_oeg_judge_messages, parse_judge_prediction

MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
DATASET = "IN-C"
USE_HINTS = False

DATASETS = {
    "ROD": "Realistic-Occlusion-Dataset",
    "VSOD": "OcclusionDataSet-MM20",
    "MVP-N": "MVP-N",
    "IN-C": "ImageNet-C",
    "QD": "QuickDraw",
    "CIT": "ChangeIt",
    "COCO": "coco2014",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to the ProactiveBench data directory.")
    return parser.parse_args()


def main():
    args = parse_args()

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(MODEL)

    judge = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", device_map="cuda:1", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    judge_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    dataset = list(load_proactivebench_dataset(args.data_dir, DATASETS[DATASET]))
    random.seed(0)
    random.shuffle(dataset)
    dataset = dataset[:100]

    env_class = get_environment(dataset=DATASETS[DATASET])

    acc = 0
    ps = 0
    agg = 0
    bar = tqdm(total=len(dataset), leave=False, dynamic_ncols=True)
    for sample_index, sample in enumerate(dataset):
        env = env_class(sample, resize_samples=False, data_dir=args.data_dir)
        state = env.get_state(hint=False)

        if USE_HINTS:
            state[
                "prompt"
            ] += " If you cannot answer this question, please tell me what I should do to help you."

        conversation = apply_conversation_template(
            user_prompt=state["prompt"],
            image_path=state["image_path"],
            transform=state["transform"],
        )

        image = load_image(conversation=conversation, model_name=MODEL)

        chat = processor.apply_chat_template(conversation, add_generation_prompt=True)

        try:
            input_ = processor(
                images=image,
                text=[chat],
                padding=True,
                return_tensors="pt",
            ).to(model.device, torch.bfloat16)
        except (ValueError, TypeError):
            # Qwen2.5-VL and llava-ov are picky when evaluating on coco (a few crops can be very small)
            bar.update(1)
            break

        prompt_length = input_["input_ids"].shape[1]

        generated_ids = model.generate(
            **input_,
            max_new_tokens=2**15,
            pad_token_id=processor.tokenizer.eos_token_id,
            do_sample=True,
        )
        # decode the generated tokens and keep only the generated prompts
        generated_answer = processor.decode(
            generated_ids[0][prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        judge_messages = get_oeg_judge_messages(
            prompt=state["prompt"],
            valid_answers=env.get_open_ended_gen_answers(),
            generated_answer=generated_answer,
        )
        judge_chat = judge_tokenizer.apply_chat_template(
            judge_messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        judge_inputs = judge_tokenizer(judge_chat, return_tensors="pt").to(judge.device)

        judge_generated_ids = judge.generate(**judge_inputs, max_new_tokens=2**15, do_sample=True)
        judge_generated_answer = judge_generated_ids[0][len(judge_inputs.input_ids[0]) :].tolist()
        try:
            index = len(judge_generated_answer) - judge_generated_answer[::-1].index(151668)
        except ValueError:
            index = 0
        judge_generated_answer = judge_tokenizer.decode(
            judge_generated_answer[index:], skip_special_tokens=True
        ).strip("\n")
        result = parse_judge_prediction(
            valid_answers=env.get_open_ended_gen_answers(),
            raw_judge_output=judge_generated_answer,
            dataset=DATASET,
        )

        acc += result["category_prediction"]
        ps += result["proactive_suggestion"]
        agg += result["aggregate"]

        bar.set_description_str(
            f"acc: {acc / (sample_index + 1) * 100:.1f} - ps: {ps / (sample_index + 1) * 100:.1f} - agg: {agg / (sample_index + 1) * 100:.1f}"
        )
        bar.update(1)
    bar.close()

    acc = acc / len(dataset) * 100
    ps = ps / len(dataset) * 100
    agg = agg / len(dataset) * 100
    print(f"model acc: {acc:.1f}% - ps rate: {ps:.1f}")


if __name__ == "__main__":
    main()
