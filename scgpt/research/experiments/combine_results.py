import json
import csv

INPUT_FILE = "scgpt/research/results/cancer_predictions/summary.json"
OUTPUT_FILE = "scgpt/research/results/cancer_predictions/summary.csv"

def json_to_csv(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    fieldnames = ["model", "query", "policy", "n", "pearson_full", "spearman_full", "pearson_low", "spearman_low", "pearson_high", "spearman_high"]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved CSV to {output_file}")

if __name__ == "__main__":
    json_to_csv(INPUT_FILE, OUTPUT_FILE)