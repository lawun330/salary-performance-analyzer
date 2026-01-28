# Insights

I learned several important concepts and techniques in this project:

### End-to-End Project Workflow
  - Choose dataset and define prediction targets
  - Select features and drop irrelevant columns
  - Check distributions and run basic EDA
  - Encode features/targets (one-hot, ordinal, label encoding)
  - Train models with a repeatable setup (pipelines in notebooks)
  - Choose the best model
  - Build a Streamlit dashboard for interactive inference and visualization

### Bidirectional Modeling (Salary ↔ Performance)
  - Given pay and profile, estimate a realistic performance level:
    $$\text{Performance} = f(\text{Salary}, \text{Profile})$$
  - Given target performance and profile, estimate a reasonable salary:
    $$\text{Salary} = f(\text{Performance}, \text{Profile})$$

### Feature Preparation
- `feature_info` is saved and used to rebuild DataFrames in the exact column order expected by the models.
- `label_encoder` is saved and reused to decode predicted classes back into the original performance score scale.
- Inference inputs must match the training schema.

### Model Persistence
- `joblib`is used to save and load scikit-learn models, encoders, and feature metadata (`feature_info`).

### ML Pipelines
- **ColumnTransformer**: Apply preprocessing steps by feature type.
- **Scikit-learn Pipeline**: Combine preprocessing and modeling into a single pipeline for consistency and reproducibility.

### Hyperparameter Tuning
- **GridSearchCV**: Entirely skipped here because the dataset is relatively deterministic and baseline performance was already stable.

### Optimization
- **Rationale**: Prediction tells what happens; optimization chooses what to do.
- **Note**: Grid search used here is not `GridSearchCV`. This grid search is a simple search over a grid of candidate salary values. The salary range is discretized into many evenly spaced points (a grid), the model is evaluated at each point, and the best point is selected based on the chosen objective.
- **Objectives**:
  - Maximize performance (budget-constrained)
  - Maximize ROI (budget-constrained)
  - Minimize salary (meet target; optional budget cap)
- **Tie-break rule**: Choose the lowest salary among best-scoring options

### Constraints and Edge Cases
- **Clipping**: Predicted salaries are clipped to a realistic range to prevent unusable recommendations.
- **Unrealistic Cases**: When a target performance cannot be reached within budget, a warning is shown with the best achievable performance.

### Streamlit Dashboard Engineering
- **Caching**: `st.cache_resource` avoids reloading artifacts on every rerun and keeps the app responsive.
- **UI**: Sidebar inputs + main results area keeps the workflow easy to follow.
- **Visualization**: Plotting a curve helps validate and explain recommendations.
