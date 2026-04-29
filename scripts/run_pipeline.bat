@echo off
REM ============================================================
REM Run: scripts/run_pipeline.bat at the root of the repo
REM Run full Tenacious Judge pipeline
REM Loads secrets from .env automatically
REM ============================================================

echo [0/5] Loading environment variables from .env...
REM Use python-dotenv inside scripts, but we can also export here if needed
for /f "tokens=1,2 delims==" %%a in (.env) do (
    set %%a=%%b
)

C:/Users/nuhamin/Documents/Tenacious/week11/The-Conversoin-Engine-Ground-Truth/src/data_prep/adversarial_cases.py

echo [1/5] Preparing datasets...
python src\data_prep\trace_tasks.py
python src\data_prep\synthetic_pairs.py
python src\data_prep\adversarial_cases.py
python src\data_prep\split_dataset.py

echo [2/5] Training Judge model...
python src\training\train_judge.py --model_name distilbert-base-uncased --data_dir data\splits --output_dir models\judge --epochs 3 --batch_size 16

echo [3/5] Evaluating Judge model...
python src\evaluation\eval_judge.py --model_dir models\judge --data_dir data\splits --output_dir reports

echo [4/5] Evaluating Baseline agent...
python src\evaluation\eval_baseline.py --data_dir data\splits --output_dir reports

echo [5/5] Generating comparison notebook outputs...
jupyter nbconvert --execute notebooks\evaluation_results.ipynb --to html --output reports\evaluation_results.html

echo ============================================================
echo Pipeline complete! Results saved in /reports
echo - judge_metrics.json
echo - baseline_metrics.json
echo - judge_confusion_matrix.png
echo - baseline_confusion_matrix.png
echo - comparison_chart.png
echo - evaluation_results.html
echo ============================================================

pause

