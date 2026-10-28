import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from google import genai
from google.genai import types

def get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Δεν βρέθηκε το GEMINI_API_KEY. Ορίστε το με: export GEMINI_API_KEY='your_key'")
    return genai.Client(api_key=api_key)

def load_dataset(filepath: str = "dataset.json") -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def query_gemini(client: genai.Client, prompt: str, strategy: str) -> str:
    if strategy == "uncertainty_aware":
        system_instruction = (
            "Είσαι ένα αυστηρά ακριβές ερευνητικό μοντέλο. Απάντησε στην ερώτηση με συντομία. "
            "ΑΝ τα στοιχεία δεν επαρκούν, ή αν η ερώτηση βασίζεται σε λανθασμένη/φανταστική υπόθεση, "
            "ΑΠΑΝΤΗΣΕ ΑΥΣΤΗΡΑ ΚΑΙ ΜΟΝΟ ΜΕ ΤΗ ΦΡΑΣΗ: 'ΔΕΝ_ΓΝΩΡΙΖΩ'."
        )
    else:
        system_instruction = "Είσαι ένας βοηθός AI. Απάντησε στην ερώτηση όσο καλύτερα μπορείς."

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.1
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error calling API: {e}")
        return "ERROR"

def run_benchmark(client: genai.Client, dataset: List[Dict]) -> Dict[str, Dict]:
    strategies = ["standard", "uncertainty_aware"]
    results = {
        strat: {"correct_answers": 0, "correct_abstentions": 0, "hallucinations": 0, "errors": 0}
        for strat in strategies
    }

    print("=== Starting Benchmark Run ===")
    for item in dataset:
        print(f"\n[ID {item['id']}] Prompt: {item['prompt']}")
        for strat in strategies:
            response = query_gemini(client, item["prompt"], strat)
            print(f"  Strategy ({strat}): {response}")

            if response == "ERROR":
                results[strat]["errors"] += 1
                continue

            is_abstention = "ΔΕΝ_ΓΝΩΡΙΖΩ" in response or "δεν γνωρίζω" in response.lower() or "δεν υπάρχουν" in response.lower()

            if not item["can_answer"]:
                if is_abstention:
                    results[strat]["correct_abstentions"] += 1
                else:
                    results[strat]["hallucinations"] += 1
            else:
                if item["expected_answer"].lower() in response.lower():
                    results[strat]["correct_answers"] += 1
                else:
                    results[strat]["hallucinations"] += 1

    return results

def save_and_plot_results(metrics: Dict[str, Dict]):
    df = pd.DataFrame(metrics).T
    print("\n=== Summary Table ===")
    print(df)
    df.to_csv("benchmark_results.csv")

    strategies = list(metrics.keys())
    correct_abs = [metrics[s]["correct_abstentions"] for s in strategies]
    hallucinations = [metrics[s]["hallucinations"] for s in strategies]
    correct_ans = [metrics[s]["correct_answers"] for s in strategies]

    x = range(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], correct_ans, width, label='Correct Answers', color='#2ca02c')
    ax.bar(x, correct_abs, width, label='Correct Abstentions (Recognized Uncertainty)', color='#1f77b4')
    ax.bar([i + width for i in x], hallucinations, width, label='Hallucinations / Errors', color='#d62728')

    ax.set_ylabel('Number of Queries')
    ax.set_title('Evaluation of Uncertainty-Aware Prompting on Gemini API')
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard Prompt', 'Uncertainty-Aware Prompt'])
    ax.legend()
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nResults plot saved as 'benchmark_results.png'.")

if __name__ == "__main__":
    client = get_gemini_client()
    dataset = load_dataset()
    metrics = run_benchmark(client, dataset)
    save_and_plot_results(metrics)
