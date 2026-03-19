import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

try:
    from proactivebench.data_utils import (
        apply_conversation_template,
        apply_multi_choice_template,
        load_image,
        load_proactivebench_dataset,
    )
    from proactivebench.environment import get_environment
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from proactivebench.data_utils import (
        apply_conversation_template,
        apply_multi_choice_template,
        load_image,
        load_proactivebench_dataset,
    )
    from proactivebench.environment import get_environment


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
        MODEL, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(MODEL)

    dataset = load_proactivebench_dataset(args.data_dir, DATASETS[DATASET])
    Environment_class = get_environment(dataset=DATASETS[DATASET])

    acc = 0
    ps = 0
    bar = tqdm(total=len(dataset), leave=False, dynamic_ncols=True)
    for sample_index, sample in enumerate(dataset):
        # create an environment for each sample
        env = Environment_class(entry=sample, data_dir=args.data_dir)

        while not env.stop:
            # set hint=True to enable hints
            state = env.get_state(hint=USE_HINTS)

            # build input
            # {question} + Choose from the following options. Options:\n
            # {options}
            # (if short answer) Please only return one of the options without any other words.
            user_prompt = apply_multi_choice_template(
                question=state["prompt"], options=state["options"], short_answer=True
            )

            # convert to conversation
            # [{"role": "user", "content": [...]}]
            conversation = apply_conversation_template(
                user_prompt=user_prompt,
                image_path=state["image_path"],
                transform=state["transform"],
            )

            # load image (path retrieved from conversation)
            # if model is a qwen model, use qwen-vl-utils
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
                break

            prompt_length = input_["input_ids"].shape[1]

            generated_ids = model.generate(
                **input_,
                max_new_tokens=50,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            # decode the generated tokens and keep only the generated prompts
            generated_answer = processor.decode(
                generated_ids[0][prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            env.evaluate_generated_answer(generated_answer)

        stats = env.get_statistics()
        acc += int(stats["correct_prediction"])
        ps += stats["num_turns"] - 1

        bar.set_description_str(
            f"acc: {acc / (sample_index + 1) * 100:.1f} - ps: {ps / (sample_index+1):.1f}"
        )
        bar.update(1)
    bar.close()

    acc = acc / len(dataset) * 100
    ps = ps / len(dataset)
    print(f"model acc: {acc:.1f}% - ps rate: {ps:.1f}")


if __name__ == "__main__":
    main()
