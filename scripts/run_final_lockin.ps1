$ErrorActionPreference = "Stop"

Write-Host "[1/4] Running tests"
python -m unittest tests.test_prl_pipeline tests.test_optimized_model tests.test_final_evaluation -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

Write-Host "[2/4] Running final nested grouped benchmark"
python src/prl/09_nested_grouped_benchmark.py
if ($LASTEXITCODE -ne 0) { throw "Nested benchmark failed" }

Write-Host "[3/4] Computing date-cluster bootstrap intervals"
python src/prl/10_evaluate_nested_results.py --bootstrap 5000
if ($LASTEXITCODE -ne 0) { throw "Nested evaluation failed" }

Write-Host "[4/4] Generating final paper tables and draft"
python src/prl/11_generate_final_paper_results.py
if ($LASTEXITCODE -ne 0) { throw "Paper-result generation failed" }

Write-Host ""
Write-Host "FINAL LOCK-IN COMPLETE"
Write-Host "Open: results\prl\final_paper\final_results_draft.md"
Write-Host "Table: results\prl\final_paper\table_final_model_comparison.csv"
