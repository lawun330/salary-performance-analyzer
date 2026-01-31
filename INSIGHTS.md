# Insights

Key concepts and techniques in this project:

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
- **How it works**: This is not `GridSearchCV`. The salary range is discretized into many evenly spaced points (a grid). One DataFrame is built with one row per point (same employee profile, varying salary); the model is run once on the whole DataFrame (batch prediction), then the best point is selected from the results. No per-point loop; one batch call per grid.
- **Objectives**:
  - Maximize performance (budget-constrained)
  - Maximize ROI (budget-constrained)
  - Minimize salary (meet target; optional budget cap)
- **Tie-break rule**: Choose the lowest salary among best-scoring options

### Case 4 Problem-Solving Logic (employer_minimize_salary)
- Search only between the minimum wage floor and the spending cap. Build a grid of salaries from cap down to floor, predict performance at each, then find the lowest salary in that range that reaches the target performance.
  - **Case 1 (target achievable)**: That lowest salary reaches the target. That salary is recommended; no warning.
  - **Case 2 (target not achievable)**: The best performance achievable within budget is lower than the target. The lowest salary that achieves that best-achievable performance is recommended. Separately, the lowest salary in the full wage range that would reach the target is computed and included in the warning (e.g. "_Need at least $X to get the target_").
- **Summary**: Target = what the employer wants. When achievable, the minimum pay that hits the target is recommended. When not achievable, the minimum pay for the best achievable performance is recommended, and the warning states how much would be needed to hit the target.

### Constraints and Edge Cases
- **Clipping**: Predicted salaries are clipped to a realistic range to prevent unusable recommendations.
- **Unrealistic Cases**: When a target performance cannot be reached within budget, a warning is shown with the best achievable performance.

### Streamlit Dashboard Engineering
- **Caching**: `st.cache_resource` avoids reloading artifacts on every rerun and keeps the app responsive.
- **UI**: Fixed section and tab-based navigation keeps the workflow easy to follow.
- **Visualization**: Plotting a curve helps validate and explain recommendations.
- **Deployment**: By pushing the project to a <u>public</u> GitHub repository and connecting it to [Streamlit Community Cloud](https://streamlit.io/cloud), the dashboard becomes accessible to anyone via a shareable web link, with zero server setup required.