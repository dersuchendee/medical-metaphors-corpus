import pandas as pd
import re
import pandas as pd
from spellchecker import SpellChecker
from openai import OpenAI
from tqdm import tqdm  # Import tqdm for progress bar

client = OpenAI(api_key='key')

import pandas as pd
import re


def llm_trial(text):
    prompt = (
        "In this task, you are asked to determine if the sentence is a metaphor or not. Respond only with **Yes** and **No**. \n\n"
        f"Question: \"{text}\"\n\n"
    )
    print("LLM Prompt:")
    print(prompt)

    response = client.chat.completions.create(
        model="o1-preview",
        messages=[
            {"role": "user", "content": "You are a metaphor expert."},
            {"role": "user", "content": prompt}
        ],
        #temperature=0,
        #max_tokens=200,
    )
    respo = response.choices[0].message.content.strip()
    print("LLM Response:", respo)
    return respo


def read_qualtrics_csv(filename):
    """
    Reads a Qualtrics CSV file that has:
      - Row 0: internal variable names (e.g., StartDate, QID1, ...)
      - Row 1: descriptive question text or labels
      - Row 2: JSON metadata (e.g., {"ImportId": ...}) that is not actual response data
      - Row 3 onward: actual survey responses.

    This function:
      1. Reads rows 0 and 1 and merges them into one header.
      2. Skips row 2 entirely.
      3. Reads the remaining rows as data.

    Returns:
      A pandas DataFrame with a clean header.
    """
    # Read first two rows for header information.
    header_rows = pd.read_csv(filename, nrows=2, header=None, encoding="utf-8")

    # Merge the two header rows.
    merged_header = []
    for short, long in zip(header_rows.iloc[0], header_rows.iloc[1]):
        short = str(short).strip() if pd.notnull(short) else ""
        long = str(long).strip() if pd.notnull(long) else ""
        if short and long:
            merged_header.append(f"{short} - {long}")
        else:
            merged_header.append(short or long)

    # Read the rest of the file, skipping rows 0, 1, and 2 (the JSON metadata row).
    df_data = pd.read_csv(filename, skiprows=3, header=None, encoding="utf-8")
    df_data.columns = merged_header
    return df_data



df3 = read_qualtrics_csv("QualtricsSurvey.csv")
df_combined = df3


# Optional: set display options.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

print("Preview of Combined DataFrame:")
print(df_combined.head())


pattern = re.compile(r"QID\d+")
q_cols = [col for col in df_combined.columns if pattern.search(col)]
df_combined[q_cols] = df_combined[q_cols].fillna("")

results = []  # To store results for each column.
total_cols = 0
correct_cols = 0

print("\n--- LLM Accuracy Check on Yes/No QID Columns ---")
for col in q_cols:
    # Get non-empty responses as lowercase strings.
    responses = df_combined[col].dropna().astype(str).str.strip().str.lower()
    responses = responses[responses != ""]

    # Only consider responses that are exactly "yes" or "no".
    yn_responses = responses[responses.isin(['yes', 'no'])]
    if yn_responses.empty:
        continue  # Skip this column if no yes/no responses exist.

    total_cols += 1
    counts = yn_responses.value_counts()

    # Determine gold standard.
    if len(counts) == 1:
        gold = counts.index[0]
    elif counts.get('yes', 0) > counts.get('no', 0):
        gold = 'yes'
    elif counts.get('no', 0) > counts.get('yes', 0):
        gold = 'no'
    else:
        gold = None  # Tie among humans.

    # Extract question text from header (assume header format "QIDx - <question text>").
    if " - " in col:
        question_text = col.split(" - ", 1)[1]
    else:
        question_text = col

    # Call the LLM function on the question text.
    llm_answer = llm_trial(question_text).strip().lower()

    # Determine if the LLM answer is considered correct.
    if gold is None:
        # If there's a tie among human responses, any answer is accepted.
        result_text = "Tie among humans; accepted."
        is_correct = True
    else:
        if llm_answer == gold:
            result_text = "LLM matches the gold standard."
            is_correct = True
        else:
            result_text = "LLM does NOT match the gold standard."
            is_correct = False

    if is_correct:
        correct_cols += 1

    # Save the results for this column.
    results.append({
        "Column": col,
        "Question": question_text,
        "HumanCounts": counts.to_dict(),
        "GoldStandard": gold if gold is not None else "Tie",
        "LLMAnswer": llm_answer,
        "Result": result_text,
        "IsCorrect": is_correct
    })

    print(f"\nColumn: {col}")
    print(f"  Human responses: {counts.to_dict()}")
    print(f"  Gold standard: {gold if gold is not None else 'Tie'}")
    print(f"  LLM answer: {llm_answer}")
    print(f"  Result: {result_text}")

# Compute final accuracy score.
if total_cols > 0:
    accuracy = (correct_cols / total_cols) * 100
    print(f"\nFinal Accuracy Score: {accuracy:.2f}% ({correct_cols} out of {total_cols} columns)")
else:
    print("\nNo yes/no QID columns were found for analysis.")

# Save LLM responses and analysis results to a CSV file.
results_df = pd.DataFrame(results)
results_df.to_csv("llm_responses_o1-preview.csv", index=False)
print("\nLLM responses and analysis have been saved to 'llm_responses_gpt4.csv'.")