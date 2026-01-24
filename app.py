import streamlit as st
import sys
import os

# add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.optimization import (
    load_models,
    employee_maximize_salary,
    employee_match_performance,
    employer_maximize_performance,
    employer_maximize_roi,
    employer_minimize_salary,
    plot_performance_curve
)

# page configuration
st.set_page_config(
    page_title="Salary-Performance Analyzer",
    page_icon="💰",
    layout="wide"
)

# title
st.title("Salary-Performance Analyzer")
st.markdown("Analyze optimal salary and performance relationships using machine learning models.")

# load models (cached for performance)
@st.cache_resource
def get_models():
    """Load models once and cache them"""
    try:
        return load_models(base_path='.')
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure you're running from the project root directory.\nError: {e}")
        st.stop()

# load models
try:
    label_encoder, performance_model, salary_model, feature_info = get_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# sidebar for navigation
st.sidebar.title("Navigation")
case = st.sidebar.radio(
    "Select Analysis Case:",
    ["Employee: Salary Recommendation",
     "Employee: Performance Recommendation",
     "Employer: Maximize Performance",
     "Employer: Maximize ROI",
     "Employer: Minimize Salary"]
)

# common employee profile inputs
st.sidebar.header("Employee Profile")
job_title = st.sidebar.selectbox(
    "Job Title",
    options=['Developer', 'Manager', 'Analyst', 'Designer', 'Engineer', 'Consultant', 'Director']
)
education_level = st.sidebar.selectbox(
    "Education Level",
    options=['High School', 'Bachelor', 'Master', 'PhD']
)

work_hours = st.sidebar.number_input("Work Hours Per Week (20-60)", min_value=20, max_value=60, value=40)
projects_handled = st.sidebar.number_input("Projects Handled (1-20)", min_value=1, max_value=20, value=5)
overtime_hours = st.sidebar.number_input("Overtime Hours (0-40)", min_value=0, max_value=40, value=5)
sick_days = st.sidebar.number_input("Sick Days (0-30)", min_value=0, max_value=30, value=2)
remote_work_frequency = st.sidebar.number_input("Remote Work Frequency (0%-100%)", min_value=0, max_value=100, value=50)
team_size = st.sidebar.number_input("Team Size (1-50)", min_value=1, max_value=50, value=8)
training_hours = st.sidebar.number_input("Training Hours (0-200)", min_value=0, max_value=200, value=50)

# create employee profile dictionary
employee_profile = {
    'Job_Title': job_title,
    'Education_Level': education_level,
    'Work_Hours_Per_Week': work_hours,
    'Projects_Handled': projects_handled,
    'Overtime_Hours': overtime_hours,
    'Sick_Days': sick_days,
    'Remote_Work_Frequency': remote_work_frequency,
    'Team_Size': team_size,
    'Training_Hours': training_hours
}

# performance label mapping
performance_options = [
    ("Lowest performance", 1),
    ("Low performance", 2),
    ("Moderate performance", 3),
    ("High performance", 4),
    ("Best performance", 5),
]

def format_performance(value):
    for label, score in performance_options:
        if score == value:
            return label
    return str(value)

# main content area
if case == "Employee: Salary Recommendation":
    st.header("Case 1: Employee - Salary Recommendation")
    st.markdown("**As an employee, find the optimal salary to request based on your performance level.**")
    
    target_choice = st.selectbox(
        "Your Performance Level",
        options=performance_options,
        format_func=lambda x: x[0]
    )
    target_performance = target_choice[1]
    
    if st.button("Calculate Recommended Salary"):
        try:
            recommended_salary = employee_maximize_salary(
                employee_profile,
                target_performance,
                salary_model,
                feature_info
            )
            
            st.success(f"### Recommended Salary: ${recommended_salary:,.2f}")
            st.info(
                f"Based on your performance level, the recommended salary is ${recommended_salary:,.2f} per month."
            )
            
        except Exception as e:
            st.error(f"Error: {e}")

elif case == "Employee: Performance Recommendation":
    st.header("Case 2: Employee - Performance Recommendation")
    st.markdown("**As an employee, find the appropriate performance level to deliver based on the offered salary.**")
    
    offered_salary = st.number_input(
        "Offered Salary ($)",
        min_value=3850,
        max_value=9000,
        value=6000,
        step=100
    )
    
    if st.button("Calculate Expected Performance"):
        try:
            recommended_performance = employee_match_performance(
                employee_profile,
                offered_salary,
                performance_model,
                label_encoder,
                feature_info
            )
            
            st.success(f"### Expected Performance: {format_performance(recommended_performance)}")
            st.info(
                f"For an offered salary of ${offered_salary:,.2f}, "
                f"the expected performance level is {format_performance(recommended_performance).lower()}."
            )
            
        except Exception as e:
            st.error(f"Error: {e}")

elif case == "Employer: Maximize Performance":
    st.header("Case 3A: Employer - Maximize Performance")
    st.markdown("**As an employer, find the salary within budget that maximizes employee performance.**")
    
    salary_budget = st.number_input(
        "Salary Budget ($)",
        min_value=3850,
        max_value=9000,
        value=7500,
        step=50
    )
    
    if st.button("Find Optimal Salary"):
        try:
            recommended_salary, expected_performance, curve = employer_maximize_performance(
                employee_profile,
                salary_budget,
                performance_model,
                label_encoder,
                feature_info
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"### Recommended Salary: ${recommended_salary:,.2f}")
            with col2:
                st.success(f"### Expected Performance: {format_performance(expected_performance)}")
            
            st.info(
                f"To maximize performance within a budget of \${salary_budget:,.2f}, "
                f"offer \${recommended_salary:,.2f} which yields {format_performance(expected_performance).lower()}."
            )
            
            # visualization
            st.subheader("Performance vs Salary Curve")
            fig = plot_performance_curve(curve, recommended_salary, expected_performance)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")

elif case == "Employer: Maximize ROI":
    st.header("Case 3B: Employer - Maximize ROI")
    st.markdown("**As an employer, find the salary within budget that maximizes ROI (performance per dollar).**")
    
    salary_budget = st.number_input(
        "Salary Budget ($)",
        min_value=3850,
        max_value=9000,
        value=7500,
        step=50
    )
    
    if st.button("Find Optimal ROI"):
        try:
            recommended_salary, expected_performance, roi, curve = employer_maximize_roi(
                employee_profile,
                salary_budget,
                performance_model,
                label_encoder,
                feature_info
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"### Recommended Salary: ${recommended_salary:,.2f}")
            with col2:
                st.success(f"### Expected Performance: {format_performance(expected_performance)}")
            with col3:
                st.success(f"### ROI: {roi:.6f}")
            
            st.info(
                f"To maximize ROI within a budget of \${salary_budget:,.2f}, "
                f"offer \${recommended_salary:,.2f} which yields {format_performance(expected_performance).lower()} "
                f"with ROI of {roi:.6f} (performance per dollar)."
            )
            
            # visualization
            st.subheader("Performance vs Salary Curve")
            fig = plot_performance_curve(curve, recommended_salary, expected_performance)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")

elif case == "Employer: Minimize Salary":
    st.header("Case 4: Employer - Minimize Salary")
    st.markdown("**As an employer, find the minimum salary required to achieve a specific performance level.**")
    
    col1, col2 = st.columns(2)
    with col1:
        target_choice = st.selectbox(
            "Specific Performance Level",
            options=performance_options,
            format_func=lambda x: x[0]
        )
        target_performance = target_choice[1]
    with col2:
        salary_budget = st.number_input(
            "Salary Budget ($) - Optional",
            min_value=3850,
            max_value=9000,
            value=None,
            step=50,
            help="Optional: Maximum salary budget constraint"
        )
    
    pay_raise = st.number_input(
        "Pay Raise Increment ($) - For Compatibility",
        min_value=100,
        max_value=1000,
        value=500,
        step=100,
        help="Parameter kept for compatibility (not actively used in grid search)"
    )
    
    if st.button("Find Minimum Salary"):
        try:
            recommended_salary, expected_performance, cost_per_perf, curve, warning_message = employer_minimize_salary(
                employee_profile,
                target_performance,
                pay_raise,
                performance_model,
                label_encoder,
                feature_info,
                salary_budget=salary_budget if salary_budget else None
            )
                        
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"### Minimum Salary: ${recommended_salary:,.2f}")
            with col2:
                st.success(f"### Expected Performance: {format_performance(expected_performance)}")
            with col3:
                st.success(f"### Cost/Performance: ${cost_per_perf:,.2f}")
            
            if len(warning_message) > 0:
                st.warning(
                    f"The current budget cannot achieve {format_performance(warning_message[0]).lower()}. "
                    f"Only {format_performance(warning_message[2]).lower()} is achievable with ${warning_message[1]:.2f}."
                )
            else:
                st.info(
                    f"To achieve {format_performance(target_performance).lower()}, "
                    f"offer a minimum salary of ${recommended_salary:,.2f}."
                )
            
            # visualization
            st.subheader("Performance vs Salary Curve")
            fig = plot_performance_curve(curve, recommended_salary, expected_performance)
            ## add target line
            ax = fig.gca()
            ax.axhline(target_performance, color='g', linestyle=':', alpha=0.8, label=f'Target: {format_performance(target_performance)}')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {e}")

# footer
st.markdown("---")
st.markdown("© 2026 La Wun Nannda")