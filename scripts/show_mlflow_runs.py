"""Display all MLflow experiments, runs, model versions, parameters, and metrics in a clean CLI table.
"""
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp_name = "India_Used_Car_Price_Prediction"
exp = client.get_experiment_by_name(exp_name)

if not exp:
    print(f"No MLflow experiment found with name '{exp_name}'.")
    exit(1)

print("=" * 80)
print(f"  MLFLOW EXPERIMENT: {exp.name} (ID: {exp.experiment_id})")
print("=" * 80)

runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"])

if not runs:
    print("No runs recorded yet.")
    exit(0)

for idx, r in enumerate(runs, 1):
    info = r.info
    data = r.data
    run_name = data.tags.get("mlflow.runName", "Unnamed Run")
    version = data.tags.get("pipeline_version", "2.2.0-ct-prod")
    stage = data.tags.get("stage", "Production Champion")
    model_type = data.tags.get("model_type", "HistGradientBoostingRegressor(log1p)")
    
    r2 = data.metrics.get("r2", None)
    mae = data.metrics.get("mae_inr", None)
    rmse = data.metrics.get("rmse_inr", None)
    acc = data.metrics.get("accuracy", None)
    
    print(f"\n[Run #{idx}] {run_name} (Run ID: {info.run_id[:8]})")
    print(f"  - Model Version / Tag: {version} | Stage: {stage} | Architecture: {model_type}")
    print(f"  - Metrics: R2 = {f'{r2:.4f}' if r2 is not None else 'N/A'} | MAE = {f'Rs. {mae:,.0f}' if mae is not None else 'N/A'} | RMSE = {f'Rs. {rmse:,.0f}' if rmse is not None else 'N/A'} | Accuracy = {f'{acc*100:.1f}%' if acc is not None else 'N/A'}")
    
    # Print Artifacts
    artifacts = client.list_artifacts(info.run_id)
    art_names = [a.path for a in artifacts]
    print(f"  - Logged Artifacts ({len(art_names)}): {', '.join(art_names[:5])}{'...' if len(art_names) > 5 else ''}")

print("\n" + "=" * 80)
