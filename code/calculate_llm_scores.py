#!/usr/bin/env python
# ------------------------------------------------------------
#  evaluate_llm_metaphor.py
#  Re-score LLM outputs against majority human judgements.
#  Now with borderline weighting for close votes.
# ------------------------------------------------------------
import glob, json, re
from pathlib import Path
import numpy as np

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report


# ------------------------------------------------------------
# Qualtrics helper
# ------------------------------------------------------------
def read_qualtrics_csv(path):
    """
    Read a Qualtrics CSV where
      row 0 = internal IDs (QID…),
      row 1 = question text,
      row 2 = metadata JSON,
      row 3+ = responses.
    Returns a DataFrame with a merged header.
    """
    head = pd.read_csv(path, nrows=2, header=None, encoding="utf-8")
    merged = [
        f"{s.strip()} - {l.strip()}" if pd.notna(s) and pd.notna(l) else (s or l)
        for s, l in zip(head.iloc[0], head.iloc[1])
    ]
    data = pd.read_csv(path, skiprows=3, header=None, encoding="utf-8")
    data.columns = merged
    return data


# ------------------------------------------------------------
# Build majority-vote lookup with confidence weights
# ------------------------------------------------------------
def build_majority_lookup(df):
    pattern = re.compile(r"QID\d+(?:_\d+)?")
    qid_cols = [c for c in df.columns if pattern.search(c)]

    lookup = {}
    for col in qid_cols:
        votes = (
            df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .loc[lambda s: s.isin(["yes", "no"])]
        )
        if votes.empty:
            continue

        yes, no = votes.value_counts().reindex(["yes", "no"], fill_value=0)
        total_votes = yes + no

        if yes > no:
            majority = "yes"
            confidence = yes / total_votes  # 0.5 to 1.0
        elif no > yes:
            majority = "no"
            confidence = no / total_votes  # 0.5 to 1.0
        else:
            majority = "tie"
            confidence = 0.5

        # Calculate weight based on confidence
        # - Perfect consensus (1.0) gets weight 1.0
        # - Bare majority (0.51) gets weight ~0.02
        # - Tie (0.5) gets weight 0.0
        if confidence > 0.5:
            weight = 2 * (confidence - 0.5)  # Maps 0.5-1.0 to 0.0-1.0
        else:
            weight = 0.0

        lookup[col] = {
            "majority": majority,
            "confidence": confidence,
            "weight": weight,
            "yes_votes": yes,
            "no_votes": no,
            "total_votes": total_votes
        }

    return lookup


# ------------------------------------------------------------
# Calculate weighted metrics
# ------------------------------------------------------------
def calculate_weighted_metrics(y_true, y_pred, sample_weights):
    """Calculate accuracy and other metrics using sample weights"""
    if not y_true:
        return {
            "weighted_accuracy": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0
        }

    # Weighted accuracy
    correct = np.array(y_true) == np.array(y_pred)
    weighted_accuracy = np.average(correct, weights=sample_weights)

    # For precision/recall, we need to handle each class
    unique_labels = list(set(y_true + y_pred))

    if len(unique_labels) <= 1:
        return {
            "weighted_accuracy": weighted_accuracy,
            "weighted_precision": weighted_accuracy,
            "weighted_recall": weighted_accuracy,
            "weighted_f1": weighted_accuracy
        }

    # Calculate weighted precision and recall manually
    precisions = []
    recalls = []
    f1s = []

    for label in unique_labels:
        # True positives with weights
        tp_mask = (np.array(y_true) == label) & (np.array(y_pred) == label)
        tp_weighted = np.sum(np.array(sample_weights)[tp_mask])

        # False positives with weights
        fp_mask = (np.array(y_true) != label) & (np.array(y_pred) == label)
        fp_weighted = np.sum(np.array(sample_weights)[fp_mask])

        # False negatives with weights
        fn_mask = (np.array(y_true) == label) & (np.array(y_pred) != label)
        fn_weighted = np.sum(np.array(sample_weights)[fn_mask])

        # Calculate precision and recall
        precision = tp_weighted / (tp_weighted + fp_weighted) if (tp_weighted + fp_weighted) > 0 else 0
        recall = tp_weighted / (tp_weighted + fn_weighted) if (tp_weighted + fn_weighted) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "weighted_accuracy": weighted_accuracy,
        "weighted_precision": np.mean(precisions),
        "weighted_recall": np.mean(recalls),
        "weighted_f1": np.mean(f1s)
    }


# ------------------------------------------------------------
# Re-score a single llm_responses file
# ------------------------------------------------------------
def rescore_file(path, majority_lookup):
    df = pd.read_csv(path)
    y_true, y_pred, sample_weights = [], [], []
    mistakes = []

    # For confidence analysis
    confidence_stats = []

    for i, row in df.iterrows():
        col = row["Column"]
        raw = str(row["LLMAnswer"]).strip().lower()
        raw = re.sub(r"[*_`]+", "", raw)
        llm_answer = re.sub(r"^\W+|\W+$", "", raw)

        vote_info = majority_lookup.get(col, {})
        gold = vote_info.get("majority", "tie")
        weight = vote_info.get("weight", 0.0)
        confidence = vote_info.get("confidence", 0.5)

        # Store confidence stats
        confidence_stats.append({
            "Column": col,
            "Confidence": confidence,
            "Weight": weight,
            "Yes_votes": vote_info.get("yes_votes", 0),
            "No_votes": vote_info.get("no_votes", 0),
            "Total_votes": vote_info.get("total_votes", 0)
        })

        if gold == "tie":
            df.at[i, "Result"] = "Tie among humans; accepted."
            df.at[i, "IsCorrect"] = True
            df.at[i, "Confidence"] = confidence
            df.at[i, "Weight"] = weight
            continue

        correct = llm_answer == gold
        if not correct:
            mistakes.append({
                "Row": i,
                "Column": col,
                "LLM_answer": llm_answer,
                "Gold": gold,
                "Confidence": confidence,
                "Weight": weight,
                "Original_response": row.get("LLMAnswer", "")
            })

        y_true.append(gold)
        y_pred.append(llm_answer)
        sample_weights.append(weight)

        df.at[i, "Result"] = (
            "LLM matches the gold standard." if correct
            else "LLM does NOT match the gold standard."
        )
        df.at[i, "IsCorrect"] = correct
        df.at[i, "Confidence"] = confidence
        df.at[i, "Weight"] = weight

    # ------------------------------------------
    # DIAGNOSTIC: what labels slipped through?
    # ------------------------------------------
    from collections import Counter
    print(f"\n=== DIAGNOSTICS for {Path(path).name} ===")
    print("Unique gold labels:", set(y_true))
    print("Label counts:", Counter(y_true))
    print("Unique LLM predictions:", set(y_pred))
    print("Union gold∪pred:", set(y_true) | set(y_pred))

    # Confidence analysis
    conf_df = pd.DataFrame(confidence_stats)
    print(f"\nConfidence distribution:")
    print(f"  Mean confidence: {conf_df['Confidence'].mean():.3f}")
    print(f"  Mean weight: {conf_df['Weight'].mean():.3f}")
    print(f"  High confidence (>0.8): {(conf_df['Confidence'] > 0.8).sum()}")
    print(f"  Borderline (0.5-0.7): {((conf_df['Confidence'] > 0.5) & (conf_df['Confidence'] < 0.7)).sum()}")
    print(f"  Ties (0.5): {(conf_df['Confidence'] == 0.5).sum()}")

    # Standard metrics (unweighted)
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        pr_weighted, rc_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        pr_per_class, rc_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=["yes", "no"]
        )

        # Custom weighted metrics based on confidence
        weighted_metrics = calculate_weighted_metrics(y_true, y_pred, sample_weights)

        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        print(f"\nConfidence-weighted metrics:")
        print(f"  Weighted accuracy: {weighted_metrics['weighted_accuracy']:.3f}")
        print(f"  Weighted precision: {weighted_metrics['weighted_precision']:.3f}")
        print(f"  Weighted recall: {weighted_metrics['weighted_recall']:.3f}")
        print(f"  Weighted F1: {weighted_metrics['weighted_f1']:.3f}")

    else:
        acc = pr_weighted = rc_weighted = f1_weighted = 0.0
        pr_macro = rc_macro = f1_macro = 0.0
        pr_per_class = rc_per_class = f1_per_class = [0.0, 0.0]
        support = [0, 0]
        weighted_metrics = {
            "weighted_accuracy": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0
        }

    # Extract individual class metrics
    labels = ["yes", "no"]
    metrics_by_class = {}
    for i, label in enumerate(labels):
        if i < len(pr_per_class):
            metrics_by_class[label] = {
                "precision": pr_per_class[i],
                "recall": rc_per_class[i],
                "f1": f1_per_class[i],
                "support": support[i] if i < len(support) else 0
            }

    # Save updated CSV with confidence info
    out_path = Path(path).with_name(Path(path).stem + "_updated.csv")
    df.to_csv(out_path, index=False)

    # Print mistakes with confidence info
    if mistakes:
        print(f"\n❌ Mistakes in file {Path(path).name}:")
        mistake_df = pd.DataFrame(mistakes)
        print(mistake_df.to_string(index=False))

        # Show mistakes by confidence level
        high_conf_mistakes = mistake_df[mistake_df['Confidence'] > 0.8]
        low_conf_mistakes = mistake_df[mistake_df['Confidence'] < 0.7]
        print(f"\nHigh-confidence mistakes (>0.8): {len(high_conf_mistakes)}")
        print(f"Low-confidence mistakes (<0.7): {len(low_conf_mistakes)}")
    else:
        print(f"\n✅ No mistakes in file {Path(path).name}.")

    return {
        "File": Path(path).name,
        "Items_evaluated": len(y_true),
        "Gold_ties": len(df) - len(y_true),
        "Mean_confidence": round(conf_df['Confidence'].mean(), 3),
        "Mean_weight": round(conf_df['Weight'].mean(), 3),
        "High_conf_items": (conf_df['Confidence'] > 0.8).sum(),
        "Borderline_items": ((conf_df['Confidence'] > 0.5) & (conf_df['Confidence'] < 0.7)).sum(),
        "Accuracy": round(acc, 3),
        "Precision_weighted": round(pr_weighted, 3),
        "Recall_weighted": round(rc_weighted, 3),
        "F1_weighted": round(f1_weighted, 3),
        "Precision_macro": round(pr_macro, 3),
        "Recall_macro": round(rc_macro, 3),
        "F1_macro": round(f1_macro, 3),
        "Confidence_weighted_acc": round(weighted_metrics['weighted_accuracy'], 3),
        "Confidence_weighted_prec": round(weighted_metrics['weighted_precision'], 3),
        "Confidence_weighted_rec": round(weighted_metrics['weighted_recall'], 3),
        "Confidence_weighted_f1": round(weighted_metrics['weighted_f1'], 3),
        "Precision_yes": round(metrics_by_class["yes"]["precision"], 3),
        "Recall_yes": round(metrics_by_class["yes"]["recall"], 3),
        "F1_yes": round(metrics_by_class["yes"]["f1"], 3),
        "Support_yes": int(metrics_by_class["yes"]["support"]),
        "Precision_no": round(metrics_by_class["no"]["precision"], 3),
        "Recall_no": round(metrics_by_class["no"]["recall"], 3),
        "F1_no": round(metrics_by_class["no"]["f1"], 3),
        "Support_no": int(metrics_by_class["no"]["support"]),
        "Overall_acc": round(df["IsCorrect"].mean(), 3),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    survey_csv = "QualtricsSurveyResponses - Scimet-survey-def_July+17,+2025_06.36.csv"
    if not Path(survey_csv).exists():
        raise FileNotFoundError(
            f"Survey file '{survey_csv}' not found. "
            "Place it in the same folder or update the path."
        )

    survey_df = read_qualtrics_csv(survey_csv)
    majority_vote = build_majority_lookup(survey_df)

    print("=== MAJORITY VOTE ANALYSIS ===")
    vote_stats = pd.DataFrame([
        {
            "Column": col,
            "Majority": info["majority"],
            "Confidence": info["confidence"],
            "Weight": info["weight"],
            "Yes_votes": info["yes_votes"],
            "No_votes": info["no_votes"],
            "Total_votes": info["total_votes"]
        }
        for col, info in majority_vote.items()
    ])

    print(f"Total items: {len(vote_stats)}")
    print(f"Ties: {(vote_stats['Majority'] == 'tie').sum()}")
    print(f"High confidence (>0.8): {(vote_stats['Confidence'] > 0.8).sum()}")
    print(f"Borderline (0.5-0.7): {((vote_stats['Confidence'] > 0.5) & (vote_stats['Confidence'] < 0.7)).sum()}")
    print(f"Mean confidence: {vote_stats['Confidence'].mean():.3f}")

    # Evaluate every llm_responses*.csv in the current folder
    summaries = []
    for fp in glob.glob("llm_responses*.csv"):
        summaries.append(rescore_file(fp, majority_vote))

    # Pretty print summary
    summary_df = (
        pd.DataFrame(summaries)
            .sort_values("Overall_acc", ascending=False)
            .reset_index(drop=True)
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print("\n" + "=" * 100)
    print("LLM-vs-Human Evaluation Summary")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    # Print focused summary with confidence weighting
    print("\n" + "=" * 100)
    print("KEY METRICS SUMMARY (including confidence weighting)")
    print("=" * 100)
    key_metrics = summary_df[[
        "File", "Items_evaluated", "Mean_confidence", "High_conf_items", "Borderline_items",
        "Accuracy", "Confidence_weighted_acc", "Recall_weighted", "Confidence_weighted_rec"
    ]]
    print(key_metrics.to_string(index=False))

    # Show impact of confidence weighting
    print("\n" + "=" * 100)
    print("CONFIDENCE WEIGHTING IMPACT")
    print("=" * 100)
    weighting_impact = summary_df[["File", "Accuracy", "Confidence_weighted_acc"]].copy()
    weighting_impact["Acc_Difference"] = weighting_impact["Confidence_weighted_acc"] - weighting_impact["Accuracy"]
    print(weighting_impact.to_string(index=False))