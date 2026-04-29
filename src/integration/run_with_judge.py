# src/integration/run_with_judge.py
#
# Pipeline integration: agent generates → judge scores → decision.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi, HfFolder, upload_file

# Load secrets
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = os.getenv("HF_REPO", "your-username/the-conversion-ground-truth")


def load_judge(model_dir="models/judge"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def judge_output(prospect_input, agent_output, tokenizer, model):
    text = f"Prospect: {prospect_input}\nAgent: {agent_output}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return pred  # 1 = good, 0 = bad

def push_results_to_hub(local_file, repo_id, path_in_repo):
    """Upload a file to HuggingFace Hub if HF_TOKEN is available."""
    if HF_TOKEN:
        api = HfApi()
        api.set_access_token(HF_TOKEN)
        upload_file(
            path_or_fileobj=local_file,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"  # or "dataset" depending on artifact
        )
        print(f"✅ Uploaded {local_file} to {repo_id}/{path_in_repo}")
    else:
        print("⚠️ No HF_TOKEN found in .env, skipping HuggingFace upload")

if __name__ == "__main__":
    tokenizer, model = load_judge()
    prospect = "Yes, Tuesday at 3 PM works for me."
    agent = "Okay, thanks for letting me know."
    score = judge_output(prospect, agent, tokenizer, model)
    print("Judge score:", score)
    
    # Save result locally
    result_file = "reports/judge_integration_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f'{{"prospect":"{prospect}","agent":"{agent}","score":{score}}}')

    # Push to HuggingFace Hub
    push_results_to_hub(result_file, HF_REPO, "integration/judge_integration_result.json")
