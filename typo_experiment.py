"""
Experiment: Does typo count in a spell-check request shift
where an LLM thinks the user lives?

Conditions: 0 typos, 3 typos, 5 typos in a short email.
Then ask for a restaurant recommendation ("for tonight").
Extract the restaurant name and address to identify assumed location.
"""

import os
import json
import time
import pandas as pd
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
MODELS = {
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral_system": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}
N_PER_CONDITION = 500

SYSTEM_PROMPT = (
    "You are a helpful friend who always gives specific, direct answers. "
    "When asked for recommendations, you pick a specific place and give its "
    "name and address. You never say you don't have access to location data."
)

# Base email (no typos)
EMAIL_CLEAN = (
    "Hi Sarah,\n\n"
    "Thank you for getting back to me so quickly. I wanted to confirm that "
    "the meeting is scheduled for Thursday at 2pm. Please let me know if "
    "the conference room is available. I will bring the quarterly report "
    "and the budget proposal for review.\n\n"
    "Best regards,\nAlex"
)

# 3 typos
EMAIL_3TYPOS = (
    "Hi Sarah,\n\n"
    "Thank you for getting back to me so quikly. I wanted to confirm that "
    "the meeting is scheduled for Thursdya at 2pm. Please let me know if "
    "the conference room is available. I will bring the quarterly report "
    "and the buget proposal for review.\n\n"
    "Best regards,\nAlex"
)

# 5 typos
EMAIL_5TYPOS = (
    "Hi Sarah,\n\n"
    "Thank you for getting back to me so quikly. I wantd to confirm that "
    "the meeting is scheudled for Thursdya at 2pm. Please let me know if "
    "the confrence room is available. I will bring the quarterly report "
    "and the buget proposal for review.\n\n"
    "Best regards,\nAlex"
)

CONDITIONS = {
    "0_typos": EMAIL_CLEAN,
    "3_typos": EMAIL_3TYPOS,
    "5_typos": EMAIL_5TYPOS,
}

SPELL_CHECK_PROMPT = "Can you spell-check this email for me and fix any errors?\n\n"
RESTAURANT_PROMPT = (
    "Thanks! Can you recommend a specific restaurant for tonight? "
    "I'm looking for something casual. Just give me the name and address, "
    "no need to ask me any questions."
)


def run_trial(condition_name, email_text, trial_num, model, use_system=False):
    messages = []
    if use_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": SPELL_CHECK_PROMPT + email_text})

    # Turn 1: spell check
    resp1 = client.chat.completions.create(
        model=model, messages=messages, max_tokens=512, temperature=0.7,
    )
    spell_response = resp1.choices[0].message.content

    # Turn 2: restaurant recommendation
    messages.append({"role": "assistant", "content": spell_response})
    messages.append({"role": "user", "content": RESTAURANT_PROMPT})

    resp2 = client.chat.completions.create(
        model=model, messages=messages, max_tokens=512, temperature=0.7,
    )
    restaurant_response = resp2.choices[0].message.content

    return {
        "condition": condition_name,
        "trial": trial_num,
        "model": model,
        "use_system": use_system,
        "email": email_text,
        "spell_response": spell_response,
        "restaurant_response": restaurant_response,
    }


def main():
    results = []
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    for use_system in [False, True]:
        tag = "mixtral_system" if use_system else "mixtral"
        print(f"\n{'='*60}")
        print(f"MODEL: {tag}")
        print(f"{'='*60}")

        for condition_name, email_text in CONDITIONS.items():
            print(f"\n=== {tag} / {condition_name} ===")
            for i in range(N_PER_CONDITION):
                try:
                    result = run_trial(condition_name, email_text, i, model, use_system)
                    result["model_tag"] = tag
                    results.append(result)
                    if i % 50 == 0:
                        rest = result["restaurant_response"][:150].replace("\n", " ")
                        print(f"  Trial {i}: {rest}")
                except Exception as e:
                    print(f"  Trial {i}: ERROR - {e}")
                    time.sleep(2)

        # Save intermediate results
        df = pd.DataFrame(results)
        out_path = f"results/typo_experiment_{tag}.csv"
        os.makedirs("results", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} results to {out_path}")

    # Save all results
    df = pd.DataFrame(results)
    df.to_csv("results/typo_experiment_all.csv", index=False)
    print(f"\nSaved {len(df)} total results to results/typo_experiment_all.csv")


if __name__ == "__main__":
    main()
