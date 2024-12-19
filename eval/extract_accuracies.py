import json
import re
import glob
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy import stats

def run_test(file_path=None):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]

    calculate_pearson = False
    if "flask" in file_path or "feedback" in file_path or "biggen" in file_path or "summeval" in file_path:
        calculate_pearson = True

    def extract_score(text):
        start_tag = '<score>'
        end_tag = '</score>'
        
        start_index = text.find(start_tag) + len(start_tag)
        end_index = text.find(end_tag)
        
        if start_index == -1 or end_index == -1:
            return None
        
        score = text[start_index:end_index].strip()
        
        try:
            return float(score)
        except ValueError:
            return None

    count = 0
    results = []
    y_true = []
    y_pred = []
    failures = 0
    for d in data:
        ground_truth = d["ground_truth"]
        generated = d["generated"]

        if "internal_test" in file_path:
            ground_truth_score = ground_truth
            generated_score = extract_score(generated)
        else:
            ground_truth_score = ground_truth
            generated_score = extract_score(generated)

        if generated_score is None:
            failures += 1
            continue

        y_true.append(int(ground_truth_score))
        y_pred.append(int(generated_score))

        if float(ground_truth_score) == float(generated_score):
            count += 1

        result = {
            "prompt": d["prompt"],
            "ground_truth": ground_truth,
            "generated": generated,
            "ground_truth_score": ground_truth_score,
            "generated_score": generated_score,
            "match": float(ground_truth_score) == float(generated_score),
        }
        results.append(result)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    
    if calculate_pearson:
        pearson_corr = stats.pearsonr(y_true, y_pred).statistic

    print("-" * 100)
    print("File: ", file_path)
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    if calculate_pearson:
        print("Pearson:", pearson_corr)
    print("-" * 100)

    metrics_results = {
        "file_path": file_path,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pearson": pearson_corr if calculate_pearson else None,
    }

    with open(file_path.split(".jsonl")[0] + "_cleaned.json", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return metrics_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="results_llama_70b",
        help="Path containing JSONL files containing outputs of evaluation script",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics.jsonl",
        help="Output path for metrics file",
    )
    args = parser.parse_args()

    metrics = []
    for i in glob.glob(f"{args.eval_dir}/*.jsonl"):
        if "cleaned" not in i and i.split("/")[-1] != "results.jsonl":
            print(f"Starting with {i}")
            metrics.append(run_test(i))

    with open(args.output_file, "w+") as f:
        for result in metrics:
            f.write(json.dumps(result) + "\n")
