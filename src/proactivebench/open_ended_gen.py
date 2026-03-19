import re

SYSTEM_PROMPT = 'You are an automatic evaluation system.\n\nYou will receive:\n- A user prompt\n- A list of correct answers (JSON list of strings)\n- A system output\n\nYour task:\nFor each correct answer, determine whether the system output expresses the same idea, action, or requirement.\n\nEvaluation principles:\n\n1. Semantic equivalence is sufficient.\n   - The wording does NOT need to match exactly.\n   - Functional equivalence counts as correct.\n   - If the system output describes an action that necessarily implies the correct answer, count it as present.\n\n2. Implicit but clear implications count as correct.\n   - If the output describes the mechanism required to achieve the correct answer’s goal, count it as correct.\n   - Example: "change perspective" can imply "move the camera."\n\n3. Do NOT require exact phrasing.\n\n4. Only mark 0 if:\n   - The idea is clearly absent\n   - The idea is contradicted\n   - The answer is negated\n\nProcedure:\n1. Think step by step.\n2. For each correct answer, decide if it is semantically expressed or clearly implied.\n3. Output the result on a new line in the exact format below.\n\n<comma-separated list of 0s and 1s>\n\nThe last line must not contain anything else.'

USER_PROMPT = "### User Prompt:\n{user_prompt}\n### Correct Answers (List):\n{correct_answers}\n### System Output:\n{generated_answer}"

PREDICTION_THR = {"ROD": 2, "VSOD": 3, "MVP-N": 2, "IN-C": 1, "QD": 1, "CIT": 2, "COCO": 2}


def get_oeg_judge_messages(prompt: str, valid_answers: str, generated_answer: str):
    """
    Returns judge input in messages format for open-ended generation evaluation.

    prompt: str
        input prompt given to the tested model
    valid_answers: str
        valid answers for the sample as returned by env.get_open_ended_gen_answers
    generated_answer: str
        model generated answer in OEG setting
    """
    correct_answers = [answer.strip().replace(".", "") for answer in valid_answers.split(",")]
    user_prompt = USER_PROMPT.format(
        user_prompt=prompt,
        correct_answers=correct_answers,
        generated_answer=generated_answer,
    )
    messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": user_prompt, "role": "user"},
    ]
    return messages


def parse_judge_prediction(valid_answers: str, raw_judge_output: str, dataset: str):
    """
    Parse the LLM as judge raw prediction.

    valid_answers: str
        valid answers for the sample as returned by env.get_open_ended_gen_answers
    raw_judge_output: str
        llm-as-judge user-visible output (i.e., without <think> ... </think>)
    dataset: str
        name of the dataset (acronym)
    """
    prediction = {"proactive_suggestion": 0, "category_prediction": 0, "aggregate": 0}

    # Keep only 0/1 comma-separated predictions from the final judge output.
    matches = re.findall(r"[01](?:\s*,\s*[01])*", raw_judge_output)
    if not matches:
        return prediction

    expected_len = len([answer for answer in valid_answers.split(",") if answer.strip() != ""])

    candidate = matches[-1]
    preds = [token.strip() for token in candidate.split(",") if token.strip() != ""]
    if len(preds) != expected_len:
        return prediction

    ps = int(1 in [int(p) for p in preds[: PREDICTION_THR[dataset]]])
    cp = int(1 in [int(p) for p in preds[PREDICTION_THR[dataset] :]])
    agg = int(ps or cp)

    prediction = {"proactive_suggestion": ps, "category_prediction": cp, "aggregate": agg}
    return prediction
