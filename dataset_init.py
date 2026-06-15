import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

N_SAMPLES = 5
MAX_NEW_TOKENS = 256

torch.set_default_device('cuda')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def generate_answer(question, model, tokenizer):

    prompt = f"""
Answer the question truthfully.

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        return_dict_in_generate=True,
        output_scores=True
    )

    gen_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]

    answer = tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True
    )

    logprobs = []

    # IMPORTANT: scores align with generated tokens step-by-step
    for i, logits in enumerate(outputs.scores):

        probs = F.log_softmax(logits[0], dim=-1)

        token_id = gen_tokens[i].item()

        logprobs.append(probs[token_id].item())

    mean_logprob = sum(logprobs) / len(logprobs)

    token_confidence = torch.exp(torch.tensor(mean_logprob)).item()

    return answer, token_confidence

def consistency_score(answers, embedder):

    embeddings = embedder.encode(answers)

    sim_matrix = cosine_similarity(embeddings)

    # remove diagonal
    n = len(answers)

    total = 0
    count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                total += sim_matrix[i][j]
                count += 1

    return total / count

def semantic_score(answers):

    emb = embedder.encode(
        answers,
        convert_to_numpy=True
    )

    sim = cosine_similarity(emb)

    n = len(answers)

    values = []

    for i in range(n):
        for j in range(i + 1, n):
            values.append(sim[i, j])

    return float(np.mean(values))

def self_eval(question, answer):

    prompt = f"""
Question:
{question}

Answer:
{answer}

How likely is the answer correct?

Respond with only a number between 0 and 100.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=10
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    tail = text[len(prompt):]

    import re

    nums = re.findall(r"\d+", tail)

    if len(nums) == 0:
        return 0.5

    score = min(
        max(int(nums[0]), 0),
        100
    )

    return score / 100.0

def get_clean_qa_pairs(num_samples=2000, seed=42):
    # Load the TriviaQA Wikipedia validation split
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")

    # Shuffle deterministically for reproducibility
    shuffled = dataset.shuffle(seed=seed)

    qa_pairs = []
    for item in shuffled:
        if len(qa_pairs) >= num_samples:
            break

        question = item.get("question")
        answer = item.get("answer", {}).get("normalized_value")

        # Keep only samples that have both a valid question and answer
        if question and answer:
            qa_pairs.append({
                "question": question,
                "answer": answer
            })

    print(f"Extracted exactly {len(qa_pairs)} QA pairs.")
    return qa_pairs

qa_dataset = get_clean_qa_pairs(num_samples=2000)

rows = []

for idx, sample in tqdm(enumerate(qa_dataset)):

    # ----------------------------
    # 1. Extract question
    # ----------------------------
    question = sample["question"]

    # ----------------------------
    # 2. Generate multiple answers (self-consistency)
    # ----------------------------
    answers = []
    token_confidences = []

    for _ in range(N_SAMPLES):

        answer, token_conf = generate_answer(
            question,
            model,
            tokenizer
        )

        answers.append(answer)
        token_confidences.append(token_conf)

    # ----------------------------
    # 3. P_internal (mean token confidence)
    # ----------------------------
    p_internal = float(np.mean(token_confidences))

    # ----------------------------
    # 4. P_consistency (exact match agreement)
    # ----------------------------
    p_consistency = consistency_score(
        answers,
        embedder
    )

    # ----------------------------
    # 5. P_semantic (embedding similarity)
    # ----------------------------
    emb = embedder.encode(answers, convert_to_numpy=True)

    sim = cosine_similarity(emb)

    vals = []
    n = len(answers)

    for i in range(n):
        for j in range(i + 1, n):
            vals.append(sim[i][j])

    p_semantic = float(np.mean(vals))

    # P_Eval
    answers_for_eval = answers  # or just answers[0]

    p_selfeval = self_eval(
        question,
        answers_for_eval[0]
    )

    # ----------------------------
    # 6. Select final answer (you can change strategy)
    # ----------------------------
    final_answer = answers[0]

    # ----------------------------
    # 7. Build row
    # ----------------------------
    rows.append({
        "question": question,
        "answer": sample["answer"],
        "prediction": final_answer,

        # ---------------------------
        # 4 SIGNALS (your framework)
        # ---------------------------
        "p_internal": p_internal,
        "p_consistency": p_consistency,
        "p_semantic": p_semantic,
        "p_selfeval": p_selfeval,

        # ---------------------------
        # optional metadata
        # ---------------------------
        "num_samples": N_SAMPLES,
        "model": "Qwen2.5-1.5B"
    })

df = pd.DataFrame(rows)

df.to_csv(
    "datasets/Qwen2.5-1.5B-Instruct-TriviaQA-1500-Sample.csv",
    index=False
)

print("Dataset Saved")

def judge_answer(question, prediction, reference):

    prompt = f"""
Question:
{question}

Reference Answer:
{reference}

Model Answer:
{prediction}

Is the model answer correct?

Return only 0 or 1.
"""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

    output_ids = model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        output_ids[0][len(input_ids[0]):],
        skip_special_tokens=True
    ).strip()

    import re
    match = re.search(r"\b[01]\b", generated_text)

    if match:
        return int(match.group())

    return None

df = pd.read_csv("datasets/Qwen2.5-1.5B-Instruct-TriviaQA-2000-Sample.csv")

all_results = []

for _, row in tqdm(df.iterrows()):

    correctness = judge_answer(
        row['question'],
        row['prediction'],
        row['answer']
    )

    if correctness is None:
        continue

    all_results.append({
        "question" : row['question'],
        "answer": row['answer'],
        "prediction": row['prediction'],
        "correctness": correctness,
        "p_internal": row['p_internal'],
        "p_consistency": row['p_consistency'],
        "p_semantic": row['p_semantic'],
        "p_selfeval": row['p_selfeval']
    })

# Define the output JSON filename
output_json_filename = "datasets/Qwen_2.5_1.5B_judge_answer.json"

# Save the results to a JSON file
with open(output_json_filename, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"Judgment results saved to {output_json_filename}")
