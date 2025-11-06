import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# Load model and explainer
@st.cache_resource
def load_model():
    with open('final_model_xgb_tuned_20251105_1034.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_explainer():
    with open('lime_explainer.dill', 'rb') as f:
        explainer = dill.load(f)
    return explainer

# Load resources
try:
    model_pipeline = load_model()
    lime_explainer = load_explainer()
    st.success("✅ Model and explainer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or explainer: {e}")
    st.stop()

# Title and description
st.title("🚗 Insurance Fraud Detection System")
st.markdown("### Predict the likelihood of fraudulent insurance claims")
st.markdown("---")

# Create input form
st.markdown("## 📋 Enter Claim Information")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

# Dictionary to store inputs
inputs = {}

with col1:
    st.markdown("#### 👤 Personal Information")
    inputs['Sex'] = st.selectbox('Sex', ['Female', 'Male'])
    inputs['MaritalStatus'] = st.selectbox('Marital Status', ['Single', 'Married', 'Widow', 'Divorced'])
    inputs['AgeOfPolicyHolder'] = st.selectbox('Age of Policy Holder', 
        ['16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65'])
    inputs['Age'] = st.number_input('Age (numeric)', min_value=0, max_value=80, value=38)
    
    st.markdown("#### 🚙 Vehicle Information")
    inputs['Make'] = st.selectbox('Vehicle Make', 
        ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 'Dodge', 
         'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'])
    inputs['VehicleCategory'] = st.selectbox('Vehicle Category', ['Sport', 'Utility', 'Sedan'])
    inputs['VehiclePrice'] = st.selectbox('Vehicle Price Range', 
        ['less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000'])
    inputs['AgeOfVehicle'] = st.selectbox('Age of Vehicle', 
        ['new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'])

with col2:
    st.markdown("#### 📅 Accident Details")
    inputs['Month'] = st.selectbox('Accident Month', 
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    inputs['WeekOfMonth'] = st.number_input('Week of Month', min_value=1, max_value=5, value=3)
    inputs['DayOfWeek'] = st.selectbox('Day of Week', 
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    inputs['AccidentArea'] = st.selectbox('Accident Area', ['Urban', 'Rural'])
    
    st.markdown("#### 📝 Claim Details")
    inputs['MonthClaimed'] = st.selectbox('Month Claimed', 
        ['0', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    inputs['WeekOfMonthClaimed'] = st.number_input('Week of Month Claimed', min_value=1, max_value=5, value=3)
    inputs['DayOfWeekClaimed'] = st.selectbox('Day of Week Claimed', 
        ['0', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    inputs['Fault'] = st.selectbox('Fault', ['Policy Holder', 'Third Party'])

with col3:
    st.markdown("#### 📄 Policy Details")
    inputs['PolicyType'] = st.selectbox('Policy Type', 
        ['Sport - Liability', 'Sport - Collision', 'Sport - All Perils',
         'Sedan - Liability', 'Sedan - Collision', 'Sedan - All Perils',
         'Utility - Liability', 'Utility - Collision', 'Utility - All Perils'])
    inputs['BasePolicy'] = st.selectbox('Base Policy', ['Liability', 'Collision', 'All Perils'])
    inputs['PolicyNumber'] = st.number_input('Policy Number', min_value=1, max_value=15420, value=7710)
    inputs['RepNumber'] = st.number_input('Rep Number', min_value=1, max_value=16, value=8)
    inputs['Deductible'] = st.number_input('Deductible', min_value=300, max_value=700, value=400)
    inputs['DriverRating'] = st.number_input('Driver Rating', min_value=1, max_value=4, value=2)
    inputs['Year'] = st.number_input('Year', min_value=1994, max_value=1996, value=1995)
    inputs['Days_Policy_Accident'] = st.selectbox('Days Policy to Accident', 
        ['none', '1 to 7', '8 to 15', '15 to 30', 'more than 30'])
    inputs['Days_Policy_Claim'] = st.selectbox('Days Policy to Claim', 
        ['none', '8 to 15', '15 to 30', 'more than 30'])
    inputs['PastNumberOfClaims'] = st.selectbox('Past Number of Claims', 
        ['none', '1', '2 to 4', 'more than 4'])
    inputs['NumberOfSuppliments'] = st.selectbox('Number of Supplements', 
        ['none', '1 to 2', '3 to 5', 'more than 5'])
    inputs['AddressChange_Claim'] = st.selectbox('Address Change to Claim', 
        ['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'])
    inputs['NumberOfCars'] = st.selectbox('Number of Cars', 
        ['1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8'])
    
    st.markdown("#### 🚨 Incident Details")
    inputs['PoliceReportFiled'] = st.selectbox('Police Report Filed', ['No', 'Yes'])
    inputs['WitnessPresent'] = st.selectbox('Witness Present', ['No', 'Yes'])
    inputs['AgentType'] = st.selectbox('Agent Type', ['External', 'Internal'])

st.markdown("---")

# Prediction button
col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
with col_pred2:
    predict_button = st.button("🔮 Predict Fraud Likelihood", use_container_width=True, type="primary")

# Make prediction
if predict_button:
    try:
        # Create dataframe from inputs
        input_df = pd.DataFrame([inputs])
        
        # Get prediction
        prediction = model_pipeline.predict(input_df)[0]
        prediction_proba = model_pipeline.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH RISK - Potential Fraud Detected")
                st.markdown(f"**Fraud Probability:** {prediction_proba[1]*100:.2f}%")
            else:
                st.success("### ✅ LOW RISK - Legitimate Claim")
                st.markdown(f"**Fraud Probability:** {prediction_proba[1]*100:.2f}%")
        
        with result_col2:
            # Create probability bar chart
            fig, ax = plt.subplots(figsize=(6, 2))
            categories = ['Legitimate', 'Fraudulent']
            probs = [prediction_proba[0]*100, prediction_proba[1]*100]
            colors = ['#28a745', '#dc3545']
            ax.barh(categories, probs, color=colors, alpha=0.7)
            ax.set_xlabel('Probability (%)')
            ax.set_xlim(0, 100)
            for i, v in enumerate(probs):
                ax.text(v + 2, i, f'{v:.2f}%', va='center')
            plt.tight_layout()
            st.pyplot(fig)
        
        # LIME Explanation
        st.markdown("---")
        st.markdown("## 🔍 Explanation (LIME)")
        st.markdown("Understanding which features contributed to this prediction:")
        
        with st.spinner('Generating explanation...'):
            # Transform input using preprocessing step
            preprocessed_data = model_pipeline.named_steps['preprocessing'].transform(input_df)
            
            # Get LIME explanation
            # Note: LIME explainer expects the predict_proba function
            explanation = lime_explainer.explain_instance(
                preprocessed_data.iloc[0].values,
                model_pipeline['Model'].predict_proba,
                num_features=10
            )
            
            # Display LIME plot
            fig = explanation.as_pyplot_figure()
            st.pyplot(fig)
            
            # Display feature importance as table
            st.markdown("### Feature Contributions")
            exp_list = explanation.as_list()
            exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution'])
            exp_df['Impact'] = exp_df['Contribution'].apply(lambda x: '🔴 Increases Fraud Risk' if x > 0 else '🟢 Decreases Fraud Risk')
            exp_df['Contribution'] = exp_df['Contribution'].apply(lambda x: f"{x:.4f}")
            st.dataframe(exp_df, use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("*This model is for demonstration purposes. Always verify results with domain experts.*")