from pathlib import Path

from backend.ml.evaluation.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_synthetic(tmp_path):
    runner = BenchmarkRunner(output_dir=tmp_path, alpha=0.10, test_size=0.25)
    report = runner.run_evaluation(
        max_samples=8,
        run_robustness=False,
        run_shuffled_control=True,
    )

    assert "summary" in report
    assert "multitask_metrics" in report
    assert "ece_details" in report
    assert (tmp_path / "benchmark_evaluation_report.json").exists()
    assert (tmp_path / "benchmark_evaluation_report.md").exists()
