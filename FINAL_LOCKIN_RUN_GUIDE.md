# Final PRL Results Lock-in

These scripts replace development-model selection with nested acquisition-date-grouped evaluation.

## Run order

```powershell
python -m unittest tests.test_prl_pipeline tests.test_optimized_model tests.test_final_evaluation -v
python src/prl/09_nested_grouped_benchmark.py
python src/prl/10_evaluate_nested_results.py --bootstrap 5000
python src/prl/11_generate_final_paper_results.py
```

## Final outputs

- `results/prl/final_nested/nested_oof_predictions.csv`
- `results/prl/final_nested/nested_model_metrics.csv`
- `results/prl/final_nested/evaluation/metric_confidence_intervals.csv`
- `results/prl/final_nested/evaluation/paired_vs_dummy.csv`
- `results/prl/final_paper/final_results_draft.md`
- `results/prl/final_paper/table_final_model_comparison.csv`

Only the nested outer-fold values should be used in the paper's abstract, results, discussion, and conclusion.
The fitted `models/prl/final_nested_*.joblib` files are deployment artifacts; their training scores are not test results.
