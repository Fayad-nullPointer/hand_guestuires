## MLflow Tracking in This Project

This project uses **MLflow** to track machine learning experiments, manage models, and log artifacts for the hand recognition task.

### Key MLflow Features Used

- **Experiment Tracking:**  
  All model training runs (Logistic Regression, Random Forest, SVM, Decision Tree, XGBoost, Stacking) are tracked using custom utility functions (`track_model`, `track_grid_search`). This logs parameters, metrics, and model artifacts for each run.

- **Artifact Logging:**  
  Evaluation artifacts such as confusion matrices are saved and logged using `add_artifact`. This allows for easy visualization and comparison of model performance.

- **Model Registry:**  
  The best models are registered and can be loaded later for inference or further evaluation using `register_model` and `load_best_model`.

- **Comparison Visualization:**  
  The `log_comparison_chart` function is used to visualize and compare the performance of different models within the MLflow UI.

### Example Usage in Code

```python
# Track a model run
track_model(model, "Model_Name", X_train, y_train, X_val, y_val, "Experiment_Name")

# Log a confusion matrix as an artifact
add_artifact(run_id="...", artifact_path="confusion_matrix.png", local_path="confusion_matrix.png")

# Load the best model from the experiment
best_model = load_best_model(experiment_name="Hand-Landmark-Recognition")
```

### Benefits

- **Reproducibility:** All experiments and results are logged for future reference.
- **Easy Comparison:** Quickly compare different models and hyperparameters.
- **Deployment Ready:** Registered models can be loaded and used in production or further research.

> **Note:** All MLflow-related utilities are imported from the custom `Mlflow` module in this project.
