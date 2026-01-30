# libraries setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# import wage bounds
from scripts.load_wage_bounds import load_wage_bounds
MINIMUM_MONTHLY_WAGE, MAXIMUM_MONTHLY_WAGE = load_wage_bounds()

########################################################################################################

# load models and feature info
## use case: label_encoder, performance_model, salary_model, feature_info = load_models(base_path='.')
def load_models(base_path=None):
    """
    Load all required models and feature info
    
    Args:
        base_path: Base directory path for model files
    
    Returns:
        label_encoder, performance_model, salary_model, feature_info
    """
    if base_path is None:
        base_path = os.getcwd()
    
    # load encoder
    label_encoder = joblib.load(os.path.join(base_path, 'preprocessors', 'label_encoder.pkl'))
    
    # load models
    performance_model = joblib.load(os.path.join(base_path, 'models', 'performance_prediction_model.pkl'))
    salary_model = joblib.load(os.path.join(base_path, 'models', 'salary_prediction_model.pkl'))
    
    # load feature info to know feature order/names
    feature_info = joblib.load(os.path.join(base_path, 'data', 'processed', 'performance_feature_info.pkl'))
    
    return label_encoder, performance_model, salary_model, feature_info

########################################################################################################

def prepare_features_for_performance_prediction(employee_profile, offered_salary, feature_info):
    """
    Prepare features DataFrame for performance prediction model
    
    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        offered_salary: monthly salary amount
        feature_info: dictionary containing feature order information
    
    Returns:
        df: DataFrame with features in correct order for performance prediction model
    """
    # create DataFrame from employee profile
    df = pd.DataFrame([employee_profile])
    
    # add salary to the DataFrame
    df['Monthly_Salary'] = offered_salary
    
    # get correct column order
    feature_order = feature_info['all_features']
    
    # reorder columns to match training data
    df = df[feature_order]
    
    return df

########################################################################################################

def prepare_features_for_salary_prediction(employee_profile, target_performance, feature_info):
    """
    Prepare features DataFrame for salary prediction model
    
    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        target_performance: target performance level (1-5)
        feature_info: dictionary containing feature order information
    
    Returns:
        df: DataFrame with features in correct order for salary prediction model
    """
    # create DataFrame from employee profile
    df = pd.DataFrame([employee_profile])
    
    # add target performance to the DataFrame
    df['Performance_Score'] = target_performance
    
    # get correct column order # replace Monthly_Salary with Performance_Score
    feature_order = [
        'Performance_Score' if c == 'Monthly_Salary' else c
        for c in feature_info['all_features']
    ]
    
    # reorder columns to match training data
    df = df[feature_order]
    
    return df

########################################################################################################

# Helper function
def predict_same_employee_performance_batch(employee_profile, salaries, performance_model, label_encoder, feature_info):
    """
    Vectorize performance prediction for many salary values (one batch call instead of N calls).

    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        salaries: 1D array of salary values to evaluate
        performance_model: trained performance prediction model
        label_encoder: label encoder for performance scores
        feature_info: dictionary containing feature order information

    Returns:
        performances: 1D array of performance scores (same length as salaries)
    """
    n = len(salaries)  # get the number of performance predictions to make
    feature_order = feature_info['all_features']  # get correct column order

    # duplicate employee profile for each salary (same profile with different Monthly_Salary)
    df = pd.DataFrame([employee_profile] * n)
    df['Monthly_Salary'] = salaries
    df = df[feature_order]  # reorder columns to match training data

    pred_encoded = performance_model.predict(df)  # predict performance for each salary
    performances = label_encoder.inverse_transform(pred_encoded)  # decode the encoded performance label back to original Performance_Score scale
    return np.asarray(performances)

########################################################################################################

# Case 1: (For Employee) Salary Recommendation Based on Target Performance
def employee_maximize_salary(employee_profile, target_performance, salary_model, feature_info, 
                            salary_range=(MINIMUM_MONTHLY_WAGE, MAXIMUM_MONTHLY_WAGE)):
    """
    Find optimal salary for employee based on their target performance level
    
    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        target_performance: desired performance level (1-5) that employee will deliver
        salary_model: trained salary prediction model
        feature_info: dictionary containing feature order information
        salary_range: tuple of (min_monthly_salary, max_monthly_salary) for validation
    
    Returns:
        recommended_salary: salary that should be requested based on target performance
    """
    
    # prepare features with DataFrame
    df = prepare_features_for_salary_prediction(employee_profile, target_performance, feature_info)
    
    # predict monthly salary
    predicted_salary = salary_model.predict(df)[0]

    # clamp/constrain predicted salary to reasonable salary range
    recommended_salary = np.clip(predicted_salary, salary_range[0], salary_range[1])
    
    return recommended_salary

########################################################################################################

# Case 2: (For Employee) Performance Recommendation Based on Offered Salary
def employee_match_performance(employee_profile, offered_salary, performance_model, label_encoder, feature_info):
    """
    Find optimal performance for employee based on offered salary
    
    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        offered_salary: monthly salary amount offered by employer
        performance_model: trained performance prediction model
        label_encoder: label encoder for performance scores
        feature_info: dictionary containing feature order information
    
    Returns:
        recommended_performance: performance level that should be delivered for this salary
    """
    
    # prepare features with DataFrame
    df = prepare_features_for_performance_prediction(employee_profile, offered_salary, feature_info)
    
    # predict performance score
    pred_encoded = performance_model.predict(df)[0]
    
    # decode the encoded performance label back to original Performance_Score scale
    recommended_performance = label_encoder.inverse_transform([pred_encoded])[0]
    
    return recommended_performance

########################################################################################################

# Case 3A: (For Employer) Performance Maximization Within Salary Budget
def employer_maximize_performance(employee_profile, salary_budget, performance_model, label_encoder, feature_info,
                                  salary_range=(MINIMUM_MONTHLY_WAGE, MAXIMUM_MONTHLY_WAGE)):
    """
    Find optimal salary for employer to maximize performance within budget
    
    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        salary_budget: maximum monthly salary budget available
        performance_model: trained performance prediction model
        label_encoder: label encoder for performance scores
        feature_info: dictionary containing feature order information
        salary_range: tuple of (min_monthly_salary, max_monthly_salary)
    
    Returns:
        recommended_salary: salary within budget that maximizes performance
        expected_performance: maximum expected performance at recommended salary
        performance_curve: array of (salary, performance) pairs for visualization
    """

    # set the bounds for optimization search
    bounds = (salary_range[0], min(salary_budget, salary_range[1]))
    
    # use vectorized grid search
    salaries_to_test = np.linspace(bounds[0], bounds[1], 200)  # more points = better coverage
    performances = predict_same_employee_performance_batch(employee_profile, salaries_to_test, performance_model, label_encoder, feature_info)
    
    # find the maximum performance
    max_performance = np.max(performances)
    max_performance_indices = np.where(performances == max_performance)[0]

    # pick the lowest salary among all salaries that give the same performance
    best_idx = max_performance_indices[0]  # first index = lowest salary
    recommended_salary = salaries_to_test[best_idx]
    expected_performance = performances[best_idx]

    # visualization
    salaries_curve = np.linspace(bounds[0], bounds[1], 100)
    performances_curve = predict_same_employee_performance_batch(employee_profile, salaries_curve, performance_model, label_encoder, feature_info)
    curve = np.column_stack([salaries_curve, performances_curve])
    
    return recommended_salary, expected_performance, curve

########################################################################################################

# Case 3B: (For Employer) Return on Investment Maximization
def employer_maximize_roi(employee_profile, salary_budget, performance_model, label_encoder, feature_info,
                         salary_range=(MINIMUM_MONTHLY_WAGE, MAXIMUM_MONTHLY_WAGE)):
    """
    Find optimal salary for employer to maximize return on investment (performance per dollar)
    
    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        salary_budget: maximum monthly salary budget available
        performance_model: trained performance prediction model
        label_encoder: label encoder for performance scores
        feature_info: dictionary containing feature order information
        salary_range: tuple of (min_monthly_salary, max_monthly_salary)
    
    Returns:
        recommended_salary: salary within budget that maximizes performance per dollar (ROI)
        expected_performance: expected performance at recommended salary
        roi: performance per dollar ratio at recommended salary
        performance_curve: array of (salary, performance) pairs for visualization
    """
    
    # set the bounds for optimization search
    bounds = (salary_range[0], min(salary_budget, salary_range[1]))

    # use vectorized grid search
    salaries_to_test = np.linspace(bounds[0], bounds[1], 200)
    performances = predict_same_employee_performance_batch(employee_profile, salaries_to_test, performance_model, label_encoder, feature_info)
    rois = performances / salaries_to_test

    # find the maximum ROI
    max_roi = np.max(rois)
    max_roi_indices = np.where(rois == max_roi)[0]

    # pick the lowest salary among all salaries that give the same ROI
    best_idx = max_roi_indices[0]  # first index = lowest salary
    recommended_salary = salaries_to_test[best_idx]
    expected_performance = performances[best_idx]
    expected_roi = rois[best_idx]
    
    # visualization
    salaries_curve = np.linspace(bounds[0], bounds[1], 100)
    performances_curve = predict_same_employee_performance_batch(employee_profile, salaries_curve, performance_model, label_encoder, feature_info)
    curve = np.column_stack([salaries_curve, performances_curve])

    return recommended_salary, expected_performance, expected_roi, curve

########################################################################################################

# Case 4: (For Employer) Salary Minimization for Target Performance
def employer_minimize_salary(employee_profile, target_performance, performance_model, label_encoder, 
                            feature_info, salary_budget=None, 
                            salary_range=(MINIMUM_MONTHLY_WAGE, MAXIMUM_MONTHLY_WAGE)):
    """
    Find minimum salary for employer to achieve target performance
    
    Args:
        employee_profile: dict with employee features (without Monthly_Salary and Performance_Score)
        target_performance: desired performance level (1-5) that employer needs (can be integer like 4.0 or fractional like 4.5)
        performance_model: trained performance prediction model
        label_encoder: label encoder for performance scores
        feature_info: dictionary containing feature order information
        salary_budget: optional maximum salary budget
        salary_range: tuple of (min_monthly_salary, max_monthly_salary) for validation
    
    Returns:
        recommended_salary: minimum salary that should be offered to achieve target performance
        expected_performance: expected performance at recommended salary
        cost_per_performance: cost per performance point
        curve: array of (salary, performance) pairs for visualization
        warning_message: Warning message string if target not achievable
    """

    # set the bounds for optimization search
    bounds = (salary_range[0], min(salary_budget or salary_range[1], salary_range[1]))
    start_salary = bounds[0]
    max_salary = bounds[1]

    # vectorized grid search
    salaries_to_test = np.linspace(start_salary, max_salary, 200)
    performances = predict_same_employee_performance_batch(employee_profile, salaries_to_test, performance_model, label_encoder, feature_info)

    # find minimum salary where performance >= target_performance
    meets_target = performances >= target_performance
    meets_indices = np.where(meets_target)[0]
    if meets_indices.size > 0:
        best_idx = meets_indices[0]
        recommended_salary = salaries_to_test[best_idx]
        expected_performance = performances[best_idx]
        warning_message = []
    else:
        # never recommend below minimum wage; recommend min wage or full budget
        recommended_salary = max(max_salary, start_salary)
        # performance at the salary we actually recommend: [0]=min wage, [-1]=full budget
        expected_performance = performances[0] if recommended_salary == start_salary else performances[-1]
        # warn if target not achievable
        warning_message = [target_performance, recommended_salary, expected_performance] if expected_performance < target_performance else []

    cost_per_performance = recommended_salary / target_performance

    # visualization
    salaries_curve = np.linspace(start_salary, max_salary, 100)
    performances_curve = predict_same_employee_performance_batch(employee_profile, salaries_curve, performance_model, label_encoder, feature_info)
    curve = np.column_stack([salaries_curve, performances_curve])

    return recommended_salary, expected_performance, cost_per_performance, curve, warning_message

########################################################################################################

# Visualization: Performance vs Salary Curve
def plot_performance_curve(performance_curve, recommended_salary=None, expected_performance=None):
    """
    Plot performance vs salary curve
    
    Returns:
        matplotlib figure object
    """
    plt.figure(figsize=(10, 6))
    plt.plot(performance_curve[:, 0], performance_curve[:, 1], 'b-', label='Expected Performance')
    
    if recommended_salary is not None:
        plt.axvline(recommended_salary, color='r', linestyle='--', 
                   label=f'Recommended Salary: ${recommended_salary:,.0f}')
        if expected_performance is not None:
            plt.plot(recommended_salary, expected_performance, 'ro', markersize=10,
                    label=f'Optimal Point: {expected_performance:.2f}')
    
    plt.xlabel('Monthly Salary ($)')
    plt.ylabel('Expected Performance Score')
    plt.title('Performance vs Salary Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

########################################################################################################