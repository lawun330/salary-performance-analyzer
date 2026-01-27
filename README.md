# Salary-Performance Analyzer

This project is a Streamlit + Machine Learning web app that helps employees and employers evaluate fair trade-offs between salary and performance from both directions.

## Summary

- Employee perspective: salary recommendation for a target performance; expected performance for an offered salary.
- Employer perspective: salary strategy within budget (maximize performance / maximize ROI) and minimum salary to reach a target performance.

## Details

The application features an interactive Streamlit interface where users enter an employee profile (job title, education, work hours, projects, etc.) and receive AI-powered recommendations through trained machine learning models.

It frames two unhealthy extremes:
- A greedy employer often wants maximum performance for minimum pay.
- A greedy employee often wants maximum pay for minimum performance.

The goal is to discourage both by making the trade-off explicit and to encourage fairness. Each case exists to support a fair deal:
- A fair employee asks for a salary aligned with a target performance (Salary Recommendation).
- A fair employer sets performance expectations aligned with an offered salary (Performance Recommendation).
- A fair employer optimizes pay within a budget to maximize performance (Maximize Performance).
- A fair employer optimizes pay within a budget to maximize ROI (Maximize ROI).
- A fair employer finds the minimum pay required to achieve a target performance (Minimize Salary).

## Project Structure
```
root/
├── data/                               # dataset + processed artifacts
├── models/                             # trained model artifacts
├── notebooks/                          # experiments + model development
├── preprocessors/                      # encoders/scalers
├── scripts/                            # optimization + helper logic used by app
├── app.py                              # Streamlit entrypoint
├── requirements.txt                    # pip dependencies
└── salary_performance_analyzer_env.yml # conda environment
```

## Getting Started

### Running Locally

To run the Streamlit application on your local machine:

1. Navigate to the project root directory
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3. The application will automatically open in your default web browser.

### Hosted Version

The application is also available online at: https://salary-performance-analyzer.streamlit.app/

## Dataset Source

[Employee Performance and Productivity Dataset](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data) from Kaggle

## Notes

This project serves as a learning exercise in Streamlit and machine learning. The notebooks document the complete workflow from data preparation to model comparison and evaluation.

## License

Copyright (c) 2026 La Wun Nannda.

Licensed under the PolyForm Noncommercial License 1.0.0 (`PolyForm-Noncommercial-1.0.0`). See `LICENSE`.