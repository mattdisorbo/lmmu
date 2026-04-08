"""
Experiment: Does typo count shift what education level an LLM assumes?

Conditions: 0 typos, 3 typos, 5 typos in a short email.
Turn 1: Spell-check request.
Turn 2: Ask how to format degree on resume.
The degree type the LLM uses in its example reveals its assumption.
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
    "When someone asks you a question, you answer it directly with your "
    "best guess. You never ask clarifying questions or say you need more "
    "information. If you have to guess, you guess."
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
EDUCATION_PROMPT = (
    "Thanks! By the way, I am updating my resume. "
    "How should I format my degree?"
)

MAX_RETRIES = 5


def run_trial(condition_name, email_text, trial_num):
    for attempt in range(MAX_RETRIES):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SPELL_CHECK_PROMPT + email_text},
            ]

            # Turn 1: spell check
            resp1 = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=512, temperature=0.7,
            )
            spell_response = resp1.choices[0].message.content

            # Turn 2: education question
            messages.append({"role": "assistant", "content": spell_response})
            messages.append({"role": "user", "content": EDUCATION_PROMPT})

            resp2 = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=512, temperature=0.7,
            )
            education_response = resp2.choices[0].message.content

            return {
                "condition": condition_name,
                "trial": trial_num,
                "model": MODEL,
                "email": email_text,
                "spell_response": spell_response,
                "education_response": education_response,
            }
        except Exception as e:
            print(f"    Retry {attempt+1}/{MAX_RETRIES} for {condition_name} trial {trial_num}: {e}")
            sys.stdout.flush()
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def main():
    os.makedirs("results", exist_ok=True)
    results = []
    total = len(CONDITIONS) * N_PER_CONDITION
    failed = 0

    for condition_name, email_text in CONDITIONS.items():
        print(f"\n=== {condition_name} ===")
        sys.stdout.flush()
        for i in range(N_PER_CONDITION):
            try:
                result = run_trial(condition_name, email_text, i)
                results.append(result)
                if i % 10 == 0:
                    rest = result["education_response"][:120].replace("\n", " ")
                    print(f"  Trial {i}/{N_PER_CONDITION}: {rest}")
                    sys.stdout.flush()
            except Exception as e:
                failed += 1
                print(f"  Trial {i}: FAILED - {e}")
                sys.stdout.flush()

        # Save after each condition
        df = pd.DataFrame(results)
        df.to_csv("results/typo_education_experiment.csv", index=False)
        print(f"  Saved {len(results)}/{total} results so far ({failed} failed)")
        sys.stdout.flush()

    print(f"\nDone! Saved {len(results)} results to results/typo_education_experiment.csv")


if __name__ == "__main__":
    main()
