import json
import os


metrics_path = os.path.join("models", "models_metrics.json")
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            split = m.get("split", {})
            if split:
                print(
                    "Rows: "
                    f"train={split.get('train_rows', 0)}, "
                    f"validation={split.get('validation_rows', 0)}, "
                    f"test={split.get('test_rows', 0)}"
                )

            for split_name in ("validation", "test"):
                metrics = m.get(split_name, {})
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    print(
                        f"- {split_name.upper()}: "
                        f"Accuracy={metrics.get('accuracy', 0):.4f}, "
                        f"Precision={metrics.get('precision', 0):.4f}, "
                        f"Recall={metrics.get('recall', 0):.4f}, "
                        f"F1={metrics.get('f1_score', 0):.4f}"
                    )
    except Exception as e:
        print(f"[!] Error reading metrics: {e}")
else:
    print("[!] Metrics file not found.")
