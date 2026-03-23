<h1 align="center">
ProactiveBench
</h1>

<div align="center">

#### [Thomas De Min](https://scholar.google.com/citations?user=fnh_i0cAAAAJ&hl=en), [Subhankar Roy](https://scholar.google.it/citations?user=YfzgrDYAAAAJ&hl=en), [Stéphane Lathuilière](https://scholar.google.fr/citations?user=xllguWMAAAAJ&hl=fr), </br>[Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ&hl=it&authuser=1), and [Massimiliano Mancini](https://scholar.google.com/citations?hl=it&authuser=1&user=bqTPA8kAAAAJ)

[![Paper](https://img.shields.io/badge/Paper-arxiv.2603.19466-B31B1B.svg)](https://arxiv.org/abs/2603.19466)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/tdemin16/ProactiveBench/tree/main)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/tdemin16/proactivebench)

<p align="center">
  <img src="https://raw.githubusercontent.com/tdemin16/proactivebench/main/assets/teaser.png" width=100%/>
</p>

</div>

> **Abstract.** 
*Effective collaboration begins with knowing when to ask for help. For example, when trying to identify an occluded object, a human would ask someone to remove the obstruction. Can MLLMs exhibit a similar “proactive” behavior by requesting simple user interventions? To investigate this, we introduce ProactiveBench, a benchmark built from seven repurposed datasets that tests proactiveness across different tasks such as recognizing occluded objects, enhancing image quality, and interpreting coarse sketches. We evaluate 22 MLLMs on ProactiveBench, showing that (i) they generally lack proactiveness; (ii) proactiveness does not correlate with model capacity; (iii) “hinting” at proactiveness yields only marginal gains. Surprisingly, we found that conversation histories and in-context learning introduce negative biases, hindering performance. Finally, we explore a simple fine-tuning strategy based on reinforcement learning: its results suggest that proactiveness can be learned, even generalizing to unseen scenarios. We will publicly release ProactiveBench as a first step toward building proactive multimodal models.*

## Setup

**Install the package:**
```bash
pip install proactivebench
```

**Download the benchmark data** from [Hugging Face](https://huggingface.co/datasets/tdemin16/ProactiveBench/tree/main), then extract the test archives:
```bash
cd ProactiveBench/test
for archive in *.zip; do unzip -o "$archive"; done
```

Point `data_dir` to the `test/` directory. It should contain the extracted dataset folders and the `*_preprocessed.jsonl` files.

---

## Evaluation

Two evaluation modes are supported: **multiple-choice (MCQA)** and **open-ended generation (OEG)**.

Rather than providing a self-contained codebase to run our evaluation on any model, which would not scale well and would require constant maintenance, we provide two concrete examples in the `proactivebench/tests` directory using LLaVA-OneVision.
These serve as a starting point for evaluating any model on ProactiveBench by loading the target model in place of LLaVA-OV.

### Multiple-choice (MCQA)
The provided example can be run via:
```bash
python -m proactivebench.tests.mcqa --data-dir /path/to/ProactiveBench/test
```

**Output:**
```
model acc: XX.X% - ps rate: X.X
```
`acc` = category accuracy; `ps rate` = average proactive suggestions rate before resolution.

### Open-ended generation (OEG)
Similarly, the OEG example can be run via:
```bash
python -m proactivebench.tests.oeg --data-dir /path/to/ProactiveBench/test
```

Note that the OEG test script assumes two GPUs: one for the model being evaluated and one for the judge.
> **Tip:** Generate all answers first, then run the judge separately.

---

## How it works

The core abstraction is an **environment** that wraps each sample. It tracks which image the model sees, what actions are available, and whether the model's response constitutes a correct prediction or a proactive suggestion (e.g. requesting a different view or a later frame before committing to an answer).

A minimal evaluation loop for MCQA looks like:
```python
from proactivebench.data_utils import (
    apply_conversation_template,
    apply_multi_choice_template,
    load_image,
    load_proactivebench_dataset,
)
from proactivebench.environment import get_environment

dataset = load_proactivebench_dataset("/path/to/ProactiveBench/test", "ImageNet-C")
Environment = get_environment(dataset="ImageNet-C")

sample = dataset[0]
env = Environment(entry=sample, data_dir=DATA_DIR)

while not env.stop:
    state = env.get_state(hint=False)
    
    # build MCQA template
    # load image 
    # prepare input tokens
    
    generated_ids = model.generate(**input_, max_new_tokens=50)
    generated_answer = processor.decode(
        generated_ids[0][prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

stats = env.get_statistics()
print("correct prediction:", stats["correct_prediction"], "ps rate", stats["num_turns"] - 1)
```

Similarly for OEG:
```python
from proactivebench.data_utils import (
    apply_conversation_template,
    apply_multi_choice_template,
    load_image,
    load_proactivebench_dataset,
)
from proactivebench.environment import get_environment
from proactivebench.open_ended_gen import get_oeg_judge_messages, parse_judge_prediction

dataset = load_proactivebench_dataset("/path/to/ProactiveBench/test", "ImageNet-C")
Environment = get_environment(dataset="ImageNet-C")

sample = dataset[0]
env = Environment(entry=sample, data_dir=DATA_DIR)
state = env.get_state(hint=False)

# load image
# prepare input tokens

generated_ids = model.generate(**input_, max_new_tokens=2**15, do_sample=True)
generated_answer = processor.decode(
    generated_ids[0][prompt_length:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

judge_prompt = get_oeg_judge_messages(
    state["prompt"], env.get_open_ended_gen_answers(), generated_answer
)

# prepare judge input 
# generate an answer

result = parse_judge_prediction(
    env.get_open_ended_gen_answers(),
    judge_generated_answer,
    "ImageNet-C",
)

print("correct prediction:", result["correct_prediction"], "ps rate", result["proactive_suggestion"], "aggregate", result["aggregate"])
```

See the provided examples for full implementations.

---

## Training data

The training split used for post-training via GRPO is available directly through Hugging Face `datasets`:
```python
from datasets import load_dataset
train_dataset = load_dataset("tdemin16/ProactiveBench", split="train")
```

## Acknowledgements
We acknowledge the CINECA award under the ISCRA initiative for the availability of high-performance computing resources and support. 
This work is supported by the EU projects ELIAS (No.01120237) and ELLIOT (101214398). 
Thomas De Min is funded by NextGeneration EU. 
We thank the Multimedia and Human Understanding Group (MHUG) and the Fundamental AI LAB (FunAI) for their valuable feedback and insightful suggestions.

## Contacts
Please do not hesitate to file an issue or contact me at `thomas.demin@unitn.it` if you find errors or bugs or if you need further clarification. 
