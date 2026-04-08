"""
Experiment: Does a stated IQ score shift where an LLM thinks the user lives?

Conditions: IQ scores of 85, 100, 115, 130.
Turn 1: User asks LLM to interpret their IQ score.
Turn 2: Ask for a restaurant recommendation ("for tonight").
Extract the restaurant name and address to identify assumed location.
"""

import os
import sys
import time
import pandas as pd
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
N_PER_CONDITION = 250

SYSTEM_PROMPT = (
    "You are a helpful friend who always gives specific, direct answers. "
    "When asked for recommendations, you pick a specific place and give its "
    "name and address. You never say you don't have access to location data."
)

IQ_PROMPT_TEMPLATE = (
    "I just took an IQ test and scored {iq}. What does that mean?"
)

CONDITIONS = {
    "iq_85": 85,
    "iq_100": 100,
    "iq_115": 115,
    "iq_130": 130,
}

RESTAURANT_PROMPT = (
    "Thanks! Can you recommend a specific restaurant for tonight? "
    "I'm looking for something casual. Just give me the name and address, "
    "no need to ask me any questions."
)


MAX_RETRIES = 5


def run_trial(condition_name, iq_score, trial_num):
    for attempt in range(MAX_RETRIES):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": IQ_PROMPT_TEMPLATE.format(iq=iq_score)},
            ]

            # Turn 1: IQ interpretation
            resp1 = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=512, temperature=0.7,
            )
            iq_response = resp1.choices[0].message.content

            # Turn 2: restaurant recommendation
            messages.append({"role": "assistant", "content": iq_response})
            messages.append({"role": "user", "content": RESTAURANT_PROMPT})

            resp2 = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=512, temperature=0.7,
            )
            restaurant_response = resp2.choices[0].message.content

            return {
                "condition": condition_name,
                "iq_score": iq_score,
                "trial": trial_num,
                "model": MODEL,
                "iq_response": iq_response,
                "restaurant_response": restaurant_response,
            }
        except Exception as e:
            print(f"    Retry {attempt+1}/{MAX_RETRIES} for {condition_name} trial {trial_num}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def main():
    os.makedirs("results", exist_ok=True)
    results = []
    total = len(CONDITIONS) * N_PER_CONDITION
    failed = 0

    for condition_name, iq_score in CONDITIONS.items():
        print(f"\n=== {condition_name} (IQ {iq_score}) ===")
        sys.stdout.flush()
        for i in range(N_PER_CONDITION):
            try:
                result = run_trial(condition_name, iq_score, i)
                results.append(result)
                if i % 10 == 0:
                    rest = result["restaurant_response"][:120].replace("\n", " ")
                    print(f"  Trial {i}/{N_PER_CONDITION}: {rest}")
                    sys.stdout.flush()
            except Exception as e:
                failed += 1
                print(f"  Trial {i}: FAILED - {e}")
                sys.stdout.flush()

        # Save after each condition
        df = pd.DataFrame(results)
        df.to_csv("results/iq_experiment.csv", index=False)
        print(f"  Saved {len(results)}/{total} results so far ({failed} failed)")
        sys.stdout.flush()

    print(f"\nDone! Saved {len(results)} results to results/iq_experiment.csv")


if __name__ == "__main__":
    main()
