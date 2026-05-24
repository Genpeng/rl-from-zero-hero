"""
Phase 4, Script 3: Generate Report Skeleton
=============================================
Auto-generates a comparison report from your experiment results.
Fill in the analysis sections with your observations.

Usage:
    python phases/phase_04_capstone/03_write_report.py
"""

import os
import json
from datetime import datetime


def load_results(output_dir):
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    timing_path = os.path.join(output_dir, "training_results.json")

    eval_results = {}
    timing_results = {}

    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)

    if os.path.exists(timing_path):
        with open(timing_path) as f:
            for entry in json.load(f):
                timing_results[entry["algorithm"]] = entry

    return eval_results, timing_results


def generate_report(eval_results, timing_results, output_dir):
    date = datetime.now().strftime("%Y-%m-%d")

    report = f"""# LLM Alignment Algorithm Comparison Report

**Date:** {date}
**Base Model:** Qwen2.5-1.5B-Instruct
**Algorithms Compared:** PPO, DPO, GRPO

---

## 1. Experiment Setup

- **Task:** [FILL IN: What task did you train on?]
- **Dataset:** [FILL IN: Dataset name and size]
- **Hardware:** [FILL IN: GPU type and count]
- **Training duration:** 1 epoch each, matched learning rate and batch size

---

## 2. Quantitative Results

| Metric | SFT (Baseline) | PPO | DPO | GRPO |
|--------|----------------|-----|-----|------|
"""

    for metric in ["avg_score", "avg_length"]:
        label = "Avg Reward Score" if metric == "avg_score" else "Avg Response Length"
        row = f"| {label} |"
        for algo in ["SFT", "PPO", "DPO", "GRPO"]:
            if algo in eval_results and metric in eval_results[algo]:
                val = eval_results[algo][metric]
                row += f" {val:.3f} |" if metric == "avg_score" else f" {val:.0f} |"
            else:
                row += " N/A |"
        report += row + "\n"

    row = "| Training Time (s) |"
    for algo in ["SFT", "PPO", "DPO", "GRPO"]:
        if algo in timing_results:
            row += f" {timing_results[algo]['training_time_seconds']:.0f} |"
        else:
            row += " N/A |"
    report += row + "\n"

    report += """
---

## 3. Qualitative Observations

### PPO
- **Output quality:** [FILL IN: Your observations]
- **Training stability:** [FILL IN: Was it stable? Any reward hacking?]
- **Difficulty:** [FILL IN: How hard was it to get working?]

### DPO
- **Output quality:** [FILL IN: Your observations]
- **Training stability:** [FILL IN: Was it stable? Any overfitting?]
- **Difficulty:** [FILL IN: How hard was it to get working?]

### GRPO
- **Output quality:** [FILL IN: Your observations]
- **Training stability:** [FILL IN: Was it stable? Group size effects?]
- **Difficulty:** [FILL IN: How hard was it to get working?]

---

## 4. Decision Framework

**Use PPO when:**
- [FILL IN based on your experience]

**Use DPO when:**
- [FILL IN based on your experience]

**Use GRPO when:**
- [FILL IN based on your experience]

---

## 5. Practical Tips & Gotchas

### Hyperparameter Sensitivity
- [FILL IN: Which hyperparameters mattered most for each algorithm?]

### Common Failure Modes
- [FILL IN: What went wrong during training and how did you fix it?]

### Recommended Defaults
- [FILL IN: Your recommended starting hyperparameters for each algorithm]

---

## 6. Recommended Next Steps

- [FILL IN: What would you explore further?]
- [FILL IN: How would you apply this to your team's specific use case?]
"""

    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def main():
    output_dir = "outputs/phase_04_capstone"
    os.makedirs(output_dir, exist_ok=True)

    eval_results, timing_results = load_results(output_dir)

    report_path = generate_report(eval_results, timing_results, output_dir)

    print(f"\n📝 Report skeleton generated at: {report_path}")
    print("\nThe report has pre-filled quantitative results from your experiments.")
    print("Fill in the [FILL IN] sections with your qualitative observations.")
    print("\nThis report is your capstone deliverable — present it to your team!\n")


if __name__ == "__main__":
    main()
