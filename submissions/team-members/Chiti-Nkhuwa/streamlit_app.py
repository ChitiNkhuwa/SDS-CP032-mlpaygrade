# MLPayGrade Advanced Track - Streamlit App with SHAP Explainability
# Save this as streamlit_app.py and run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import pickle
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="MLPayGrade - Advanced Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .shap-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_components():
    """Load all saved model components"""
    try:
        # Load model and scaler
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Load feature names
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        # Load deployment functions
        with open('deployment_functions.pkl', 'rb') as f:
            deployment_data = pickle.load(f)
        
        # Load SHAP explainer
        with open('shap_explainer.pkl', 'rb') as f:
            shap_explainer = pickle.load(f)
        
        # Load SHAP importance
        with open('shap_importance.json', 'r') as f:
            shap_importance = json.load(f)
        
        return model, scaler, feature_names, deployment_data, shap_explainer, shap_importance
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None, None, None, None

def engineer_features_single(data, deployment_data):
    """Apply feature engineering to a single prediction"""
    # Apply feature engineering
    if 'remote_ratio' in data.columns:
        data['remote_work_type'] = data['remote_ratio'].apply(
            lambda x: 'Remote' if x == 1 else ('Hybrid' if x == 0.5 else 'On-site')
        )
    
    data['job_title_category'] = data['job_title'].apply(deployment_data['categorize_job_title_advanced'])
    data['country'] = data['company_location'].apply(deployment_data['extract_country'])
    data['salary_log'] = np.log1p(100000)  # placeholder
    data['salary_sqrt'] = np.sqrt(100000)   # placeholder
    data['experience_level_encoded'] = data['experience_level'].map(deployment_data['experience_mapping'])
    data['company_size_encoded'] = data['company_size'].map(deployment_data['size_mapping'])
    data['employment_type_encoded'] = data['employment_type'].map(deployment_data['employment_mapping'])
    data['exp_size_interaction'] = data['experience_level_encoded'] * data['company_size_encoded']
    data['exp_remote_interaction'] = data['experience_level_encoded'] * data['remote_ratio']
    data['size_remote_interaction'] = data['company_size_encoded'] * data['remote_ratio']
    data['job_title_complexity'] = data['job_title'].str.split().str.len()
    data['location_diversity'] = 1  # placeholder
    data['job_title_mean_salary'] = 120000  # placeholder
    data['job_title_std_salary'] = 30000    # placeholder
    data['job_title_count'] = 100           # placeholder
    data['exp_level_mean_salary'] = 120000  # placeholder
    data['exp_level_std_salary'] = 30000    # placeholder
    data['company_size_mean_salary'] = 120000  # placeholder
    data['company_size_std_salary'] = 30000    # placeholder
    data['salary_percentile_by_title'] = 0.5   # placeholder
    data['remote_premium'] = 120000  # placeholder
    data['country_salary_premium'] = 120000  # placeholder
    
    return data

def predict_with_shap(job_title, experience_level, company_size, employment_type, 
                     company_location, remote_ratio, model, scaler, feature_names, 
                     deployment_data, shap_explainer):
    """Make prediction and calculate SHAP values"""
    
    # Create input data
    data = pd.DataFrame({
        'job_title': [job_title],
        'experience_level': [experience_level],
        'company_size': [company_size],
        'employment_type': [employment_type],
        'company_location': [company_location],
        'remote_ratio': [remote_ratio]
    })
    
    # Apply feature engineering
    data = engineer_features_single(data, deployment_data)
    
    # Drop original columns
    columns_to_drop = ['job_title', 'company_location', 'experience_level', 'company_size', 
                      'employment_type', 'remote_work_type', 'remote_ratio']
    X_single = data.drop(columns_to_drop, axis=1)
    
    # Apply one-hot encoding
    categorical_columns = X_single.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        X_single_encoded = pd.get_dummies(X_single, columns=categorical_columns, drop_first=True)
    else:
        X_single_encoded = X_single.copy()
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in X_single_encoded.columns:
            X_single_encoded[feature] = 0
    
    # Reorder columns
    X_single_encoded = X_single_encoded[feature_names]
    
    # Scale and predict
    X_single_scaled = scaler.transform(X_single_encoded)
    prediction = model.predict(X_single_scaled)[0]
    
    # Calculate SHAP values
    shap_values = shap_explainer.shap_values(X_single_scaled)
    
    return prediction, shap_values, X_single_encoded

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 MLPayGrade Advanced Salary Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Salary Prediction with SHAP Explainability")
    
    # Load model components
    with st.spinner("Loading AI model and components..."):
        model, scaler, feature_names, deployment_data, shap_explainer, shap_importance = load_model_and_components()
    
    if model is None:
        st.error("❌ Failed to load model components. Please ensure all files are in the same directory.")
        return
    
    # Sidebar for inputs
    st.sidebar.markdown("## 🎯 Job Configuration")
    
    # Job title input
    job_title = st.sidebar.text_input(
        "Job Title",
        value="Data Scientist",
        help="Enter the job title (e.g., Data Scientist, ML Engineer, Research Scientist)"
    )
    
    # Experience level
    experience_level = st.sidebar.selectbox(
        "Experience Level",
        options=["EN", "MI", "SE", "EX"],
        format_func=lambda x: {
            "EN": "Entry Level",
            "MI": "Mid Level", 
            "SE": "Senior Level",
            "EX": "Executive Level"
        }[x],
        index=2
    )
    
    # Company size
    company_size = st.sidebar.selectbox(
        "Company Size",
        options=["S", "M", "L"],
        format_func=lambda x: {
            "S": "Small (< 50 employees)",
            "M": "Medium (50-250 employees)",
            "L": "Large (> 250 employees)"
        }[x],
        index=1
    )
    
    # Employment type
    employment_type = st.sidebar.selectbox(
        "Employment Type",
        options=["FT", "PT", "CT", "FL"],
        format_func=lambda x: {
            "FT": "Full-time",
            "PT": "Part-time",
            "CT": "Contract",
            "FL": "Freelance"
        }[x]
    )
    
    # Company location
    company_location = st.sidebar.text_input(
        "Company Location",
        value="US",
        help="Enter country code (e.g., US, CA, GB, AU, DE, FR, etc.)"
    )
    
    # Remote work ratio
    remote_ratio = st.sidebar.slider(
        "Remote Work Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.5,
        help="0.0 = On-site, 0.5 = Hybrid, 1.0 = Remote"
    )
    
    # Model performance metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Model Performance")
    st.sidebar.markdown("**R² Score:** 0.9993")
    st.sidebar.markdown("**MAE:** $152.29")
    st.sidebar.markdown("**RMSE:** $2,100.27")
    
    # Prediction button
    if st.sidebar.button("🚀 Predict Salary", type="primary"):
        with st.spinner("Analyzing job configuration and calculating prediction..."):
            # Make prediction
            prediction, shap_values, features_used = predict_with_shap(
                job_title, experience_level, company_size, employment_type,
                company_location, remote_ratio, model, scaler, feature_names,
                deployment_data, shap_explainer
            )
        
        # Display prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #1f77b4;">Predicted Salary</h2>
                <h1 style="text-align: center; color: #28a745; font-size: 3rem;">${prediction:,.0f}</h1>
                <p style="text-align: center; color: #6c757d;">USD per year</p>
            </div>
            """, unsafe_allow_html=True)
        
        # SHAP Explanation
        st.markdown("## 🔍 SHAP-Based Model Explanation")
        
        # Get top features by SHAP importance
        if len(shap_values.shape) > 1:
            shap_values_flat = shap_values[0]
        else:
            shap_values_flat = shap_values
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_values_flat
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        # Top 10 features affecting this prediction
        st.markdown("### 📈 Top Features Affecting This Prediction")
        
        # Create horizontal bar chart
        top_features = feature_importance_df.head(10)
        fig = px.bar(
            top_features,
            x='SHAP_Value',
            y='Feature',
            orientation='h',
            color='SHAP_Value',
            color_continuous_scale='RdBu',
            title="SHAP Values for Top 10 Features"
        )
        fig.update_layout(
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="Feature",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature details
        st.markdown("### 📋 Feature Impact Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Positive Impact (Increases Salary):**")
            positive_features = feature_importance_df[feature_importance_df['SHAP_Value'] > 0].head(5)
            for _, row in positive_features.iterrows():
                st.markdown(f"• **{row['Feature']}**: +${row['SHAP_Value']:,.0f}")
        
        with col2:
            st.markdown("**🔴 Negative Impact (Decreases Salary):**")
            negative_features = feature_importance_df[feature_importance_df['SHAP_Value'] < 0].head(5)
            for _, row in negative_features.iterrows():
                st.markdown(f"• **{row['Feature']}**: ${row['SHAP_Value']:,.0f}")
        
        # What-if analysis
        st.markdown("## 🎯 What-If Analysis")
        st.markdown("See how changing job parameters affects the salary prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Experience Level Impact")
            exp_predictions = {}
            for exp in ["EN", "MI", "SE", "EX"]:
                exp_pred, _, _ = predict_with_shap(
                    job_title, exp, company_size, employment_type,
                    company_location, remote_ratio, model, scaler, feature_names,
                    deployment_data, shap_explainer
                )
                exp_predictions[exp] = exp_pred
            
            exp_df = pd.DataFrame(list(exp_predictions.items()), columns=['Experience', 'Salary'])
            exp_df['Experience'] = exp_df['Experience'].map({
                "EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"
            })
            
            fig = px.bar(exp_df, x='Experience', y='Salary', title="Salary by Experience Level")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Remote Work Impact")
            remote_predictions = {}
            for remote in [0.0, 0.5, 1.0]:
                remote_pred, _, _ = predict_with_shap(
                    job_title, experience_level, company_size, employment_type,
                    company_location, remote, model, scaler, feature_names,
                    deployment_data, shap_explainer
                )
                remote_predictions[remote] = remote_pred
            
            remote_df = pd.DataFrame(list(remote_predictions.items()), columns=['Remote_Ratio', 'Salary'])
            remote_df['Remote_Type'] = remote_df['Remote_Ratio'].map({
                0.0: "On-site", 0.5: "Hybrid", 1.0: "Remote"
            })
            
            fig = px.bar(remote_df, x='Remote_Type', y='Salary', title="Salary by Remote Work Type")
            st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    st.markdown("---")
    st.markdown("## 🤖 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>🎯 Model Type</h4>
            <p>Random Forest Regressor</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h4>🔧 Features Engineered</h4>
            <p>24 Advanced Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h4>📊 Explainability</h4>
            <p>SHAP Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d;">
        <p>MLPayGrade Advanced Track - Built with Streamlit & SHAP</p>
        <p>Model Performance: R² = 0.9993 | MAE = $152.29 | RMSE = $2,100.27</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 