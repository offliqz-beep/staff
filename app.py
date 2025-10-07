# -*- coding: utf-8 -*-
"""
Tenant Rent Prediction & Payment Tracking System
Author: GODSON
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
def create_sample_model():
    """Create a sample model if none exists."""
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model.joblib")
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'BHK': np.random.randint(1, 6, n_samples),
        'Size': np.random.randint(500, 3000, n_samples),
        'Bathroom': np.random.randint(1, 4, n_samples),
        'Furnishing Status': np.random.choice(['Unfurnished', 'Semi-Furnished', 'Furnished'], n_samples),
        'Tenant Preferred': np.random.choice(['Bachelors', 'Family', 'Bachelors/Family'], n_samples),
        'City': np.random.choice(['Kigali', 'Musanze', 'Huye', 'Rubavu'], n_samples),
        'Point of Contact': np.random.choice(['Contact Owner', 'Contact Agent', 'Contact Builder'], n_samples),
        'Area Type': np.random.choice(['Super Area', 'Carpet Area', 'Built Area'], n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Encode categorical variables
    le_furnishing = LabelEncoder()
    le_tenant = LabelEncoder()
    le_city = LabelEncoder()
    le_contact = LabelEncoder()
    le_area = LabelEncoder()
    
    df['Furnishing Status'] = le_furnishing.fit_transform(df['Furnishing Status'])
    df['Tenant Preferred'] = le_tenant.fit_transform(df['Tenant Preferred'])
    df['City'] = le_city.fit_transform(df['City'])
    df['Point of Contact'] = le_contact.fit_transform(df['Point of Contact'])
    df['Area Type'] = le_area.fit_transform(df['Area Type'])
    
    # Create target variable (rent) based on features
    base_rent = (
        df['BHK'] * 15000 +
        df['Size'] * 50 +
        df['Bathroom'] * 8000 +
        df['Furnishing Status'] * 10000 +
        df['City'] * 5000 +
        np.random.normal(0, 5000, n_samples)
    )
    
    # Train model
    X = df[['BHK', 'Size', 'Bathroom', 'Furnishing Status', 'Tenant Preferred', 
            'City', 'Point of Contact', 'Area Type']]
    y = base_rent
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    model_data = {
        'model': model,
        'encoders': {
            'furnishing': le_furnishing,
            'tenant': le_tenant,
            'city': le_city,
            'contact': le_contact,
            'area': le_area
        }
    }
    
    joblib.dump(model_data, MODEL_FILE)
    return model_data

def load_model():
    """Load the ML model or create a sample one."""
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model.joblib")
    
    if not os.path.exists(MODEL_FILE):
        st.info("🔄 Creating a sample machine learning model...")
        return create_sample_model()
    
    try:
        model_data = joblib.load(MODEL_FILE)
        return model_data
    except Exception as e:
        st.warning(f"Could not load existing model: {e}. Creating a new sample model...")
        return create_sample_model()

def load_history():
    """Load existing history CSV or create empty DataFrame."""
    HISTORY_FILE = os.path.join(os.path.dirname(__file__), "tenant_history.csv")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
    else:
        history_df = pd.DataFrame(columns=[
            'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
            'City','Point of Contact','Area Locality','Posted On','Area Type',
            'Floor','Predicted Rent', 'Actual Rent', 'Tenant Name', 'Phone Number',
            'Contract Start', 'Contract End', 'Payment Method', 'Security Deposit'
        ])
    return history_df, HISTORY_FILE

def load_payment_history():
    """Load payment records or create empty DataFrame."""
    PAYMENT_FILE = os.path.join(os.path.dirname(__file__), "payment_history.csv")
    if os.path.exists(PAYMENT_FILE):
        payment_df = pd.read_csv(PAYMENT_FILE)
        if 'Payment Date' in payment_df.columns:
            payment_df['Payment Date'] = pd.to_datetime(payment_df['Payment Date'])
        if 'Due Date' in payment_df.columns:
            payment_df['Due Date'] = pd.to_datetime(payment_df['Due Date'])
    else:
        payment_df = pd.DataFrame(columns=[
            'Tenant Name', 'Phone Number', 'Payment Date', 'Due Date', 
            'Amount', 'Payment Method', 'Status', 'Late Days', 'Receipt Number'
        ])
    return payment_df, PAYMENT_FILE

def preprocess_input(input_data, encoders):
    """Preprocess input data for prediction."""
    processed_data = input_data.copy()
    
    # Encode categorical variables
    try:
        processed_data['Furnishing Status'] = encoders['furnishing'].transform([input_data['Furnishing Status'].iloc[0]])[0]
        processed_data['Tenant Preferred'] = encoders['tenant'].transform([input_data['Tenant Preferred'].iloc[0]])[0]
        processed_data['City'] = encoders['city'].transform([input_data['City'].iloc[0]])[0]
        processed_data['Point of Contact'] = encoders['contact'].transform([input_data['Point of Contact'].iloc[0]])[0]
        processed_data['Area Type'] = encoders['area'].transform([input_data['Area Type'].iloc[0]])[0]
    except ValueError as e:
        # If there's a new category, use the first available category
        st.warning("Some categories were not in training data. Using default encoding.")
        processed_data['Furnishing Status'] = 0
        processed_data['Tenant Preferred'] = 0
        processed_data['City'] = 0
        processed_data['Point of Contact'] = 0
        processed_data['Area Type'] = 0
    
    return processed_data[['BHK', 'Size', 'Bathroom', 'Furnishing Status', 'Tenant Preferred', 
                          'City', 'Point of Contact', 'Area Type']]

# -------------------------------
def main():
    # Page config
    st.set_page_config(page_title="Rent Prediction & Payment System", page_icon="🏠", layout="wide")

    # Load model and history
    model_data = load_model()
    model = model_data['model']
    encoders = model_data['encoders']
    
    history_df, HISTORY_FILE = load_history()
    payment_df, PAYMENT_FILE = load_payment_history()

    # App title
    st.markdown(
        """
        <h1 style="text-align:center; color:#2E86C1;">🏡 Tenant Rent Prediction & Payment System</h1>
        <p style="text-align:center; color:#7D3C98; font-size:18px;">
        Predict monthly rent, manage tenant records, and track payment history.
        </p>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Go to", ["Rent Prediction", "Tenant Management", "Payment Tracking", "Analytics Dashboard"])

    if page == "Rent Prediction":
        show_rent_prediction(model, encoders, history_df, HISTORY_FILE)
    elif page == "Tenant Management":
        show_tenant_management(history_df, HISTORY_FILE)
    elif page == "Payment Tracking":
        show_payment_tracking(history_df, payment_df, PAYMENT_FILE)
    elif page == "Analytics Dashboard":
        show_analytics_dashboard(history_df, payment_df)

def show_rent_prediction(model, encoders, history_df, HISTORY_FILE):
    """Rent prediction page."""
    st.header("🔮 Rent Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        BHK = st.number_input("BHK", min_value=1, max_value=10, value=2, key="pred_bhk")
        Size = st.number_input("Size (sq.ft)", min_value=100, max_value=10000, value=950, key="pred_size")
        Bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, key="pred_bath")
        Furnishing_Status = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"], key="pred_furnish")
        Area_Type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"], key="pred_area_type")
        Floor = st.text_input("Floor (e.g. '5 out of 10')", "5 out of 10", key="pred_floor")
    
    with col2:
        st.subheader("Tenant & Location")
        Tenant_Preferred = st.selectbox("Tenant Preferred", ["Bachelors", "Family", "Bachelors/Family"], key="pred_tenant")
        City = st.text_input("City", "Kigali", key="pred_city")
        Point_of_Contact = st.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"], key="pred_contact")
        Area_Locality = st.text_input("Area Locality", "Nyarugenge", key="pred_locality")
        Posted_On = st.date_input("Posted On", value=date.today(), key="pred_posted")
        
        # Additional tenant info for management
        st.subheader("Tenant Details (Optional)")
        tenant_name = st.text_input("Tenant Name", "John Doe", key="pred_name")
        phone_number = st.text_input("Phone Number", "+25078XXXXXXX", key="pred_phone")
        actual_rent = st.number_input("Actual Rent (if different)", min_value=0, value=0, key="pred_actual")

    # Prepare input
    input_data = pd.DataFrame({
        'BHK':[BHK],
        'Size':[Size],
        'Bathroom':[Bathroom],
        'Furnishing Status':[Furnishing_Status],
        'Tenant Preferred':[Tenant_Preferred],
        'City':[City],
        'Point of Contact':[Point_of_Contact],
        'Area Locality':[Area_Locality],
        'Posted On':[str(Posted_On)],
        'Area Type':[Area_Type],
        'Floor':[Floor]
    })

    # Predict button
    if st.button("🎯 Predict Rent", use_container_width=True):
        try:
            # Preprocess and predict
            processed_input = preprocess_input(input_data, encoders)
            predicted_rent = model.predict(processed_input)[0]
            
            # Ensure rent is reasonable
            predicted_rent = max(50000, min(500000, predicted_rent))

            # Save to history with additional tenant info
            input_data['Predicted Rent'] = predicted_rent
            input_data['Actual Rent'] = actual_rent if actual_rent > 0 else predicted_rent
            input_data['Tenant Name'] = tenant_name
            input_data['Phone Number'] = phone_number
            input_data['Contract Start'] = str(date.today())
            input_data['Contract End'] = str(date.today().replace(year=date.today().year + 1))
            input_data['Payment Method'] = "Mobile Money"
            input_data['Security Deposit'] = (actual_rent if actual_rent > 0 else predicted_rent) * 2
            
            history_df = pd.concat([history_df, input_data], ignore_index=True)
            history_df.to_csv(HISTORY_FILE, index=False)

            # Show prediction
            st.markdown(
                f"""
                <div style="background-color:#D6EAF8; padding:20px; border-radius:12px;">
                    <h2 style="color:#154360;">💰 Predicted Monthly Rent: RWF {predicted_rent:,.0f}</h2>
                    <p style="color:#2E86C1; font-size:16px;">
                    <strong>Recommended Security Deposit:</strong> RWF {(actual_rent if actual_rent > 0 else predicted_rent) * 2:,.0f} (2 months rent)
                    </p>
                </div>
                """, unsafe_allow_html=True
            )

            # Payment advice
            col_advice1, col_advice2 = st.columns(2)
            with col_advice1:
                if predicted_rent > 150000:
                    st.error("""
                    **High Rent Alert:**
                    - Consider flexible payment plans
                    - Require 2-3 months security deposit
                    - Verify tenant income stability
                    """)
                else:
                    st.success("""
                    **Reasonable Rent:**
                    - Standard payment terms applicable
                    - Tenant likely to pay on time
                    - 1-2 months security deposit recommended
                    """)
            
            with col_advice2:
                st.info("""
                **Payment Recommendations for Rwanda:**
                - ✅ Mobile Money (Momo, Airtel Money)
                - ✅ Bank Transfer
                - ✅ Cash (with signed receipt)
                - ❌ Informal cash payments
                """)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your input values and try again.")

def show_tenant_management(history_df, HISTORY_FILE):
    """Tenant management page."""
    st.header("👥 Tenant Management")
    
    if not history_df.empty:
        # Display current tenants
        st.subheader("Current Tenants")
        
        # Clean the data for display
        display_df = history_df.copy()
        if 'Actual Rent' in display_df.columns:
            display_df['Actual Rent'] = display_df['Actual Rent'].fillna(display_df['Predicted Rent'])
        
        display_columns = ['Tenant Name', 'Phone Number', 'City', 'Area Locality', 
                         'Predicted Rent', 'Actual Rent', 'Contract Start', 'Payment Method']
        available_columns = [col for col in display_columns if col in display_df.columns]
        
        st.dataframe(display_df[available_columns], use_container_width=True)
        
        # Tenant actions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Update Tenant Info")
            if len(history_df) > 0:
                tenant_index = st.selectbox("Select Tenant", range(len(history_df)), 
                                          format_func=lambda x: f"{history_df.iloc[x].get('Tenant Name', 'Unknown')} - {history_df.iloc[x].get('Phone Number', 'No Phone')}")
                
                if st.button("Update Payment Method"):
                    new_method = st.selectbox("Payment Method", 
                                            ["Mobile Money", "Bank Transfer", "Cash", "Other"])
                    history_df.at[tenant_index, 'Payment Method'] = new_method
                    history_df.to_csv(HISTORY_FILE, index=False)
                    st.success(f"Updated payment method for {history_df.iloc[tenant_index].get('Tenant Name', 'Unknown')}")
        
        with col2:
            st.subheader("Delete Tenant Record")
            if len(history_df) > 0:
                delete_index = st.number_input("Row index to delete", min_value=0, 
                                             max_value=len(history_df)-1, step=1, key="delete_idx")
                if st.button("🗑️ Delete Tenant Record", type="secondary"):
                    tenant_name = history_df.iloc[delete_index].get('Tenant Name', 'Unknown')
                    history_df.drop(index=delete_index, inplace=True)
                    history_df.reset_index(drop=True, inplace=True)
                    history_df.to_csv(HISTORY_FILE, index=False)
                    st.success(f"Record for {tenant_name} deleted successfully.")
    else:
        st.info("No tenant records yet. Go to 'Rent Prediction' to add tenants.")

def show_payment_tracking(history_df, payment_df, PAYMENT_FILE):
    """Payment tracking page."""
    st.header("💳 Payment Tracking")
    
    if not history_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Record New Payment")
            tenant_options = []
            for i in range(len(history_df)):
                tenant_name = history_df.iloc[i].get('Tenant Name', f'Tenant {i}')
                actual_rent = history_df.iloc[i].get('Actual Rent', history_df.iloc[i].get('Predicted Rent', 0))
                tenant_options.append((i, f"{tenant_name} - RWF {actual_rent:,.0f}"))
            
            if tenant_options:
                selected_tenant = st.selectbox("Select Tenant", 
                                             options=range(len(tenant_options)),
                                             format_func=lambda x: tenant_options[x][1],
                                             key="payment_tenant")
                
                tenant_data = history_df.iloc[selected_tenant]
                due_date = st.date_input("Due Date", value=date.today().replace(day=1))
                payment_date = st.date_input("Payment Date", value=date.today())
                
                default_amount = float(tenant_data.get('Actual Rent', tenant_data.get('Predicted Rent', 0)))
                amount = st.number_input("Amount Paid", value=default_amount)
                payment_method = st.selectbox("Payment Method", 
                                            ["Mobile Money", "Bank Transfer", "Cash", "Other"])
                
                # Calculate late days
                late_days = max(0, (payment_date - due_date).days)
                status = "On Time" if late_days == 0 else "Late"
                
                if st.button("💾 Record Payment", use_container_width=True):
                    new_payment = pd.DataFrame({
                        'Tenant Name': [tenant_data.get('Tenant Name', 'Unknown')],
                        'Phone Number': [tenant_data.get('Phone Number', 'Unknown')],
                        'Payment Date': [payment_date],
                        'Due Date': [due_date],
                        'Amount': [amount],
                        'Payment Method': [payment_method],
                        'Status': [status],
                        'Late Days': [late_days],
                        'Receipt Number': [f"RCPT{datetime.now().strftime('%Y%m%d%H%M%S')}"]
                    })
                    
                    payment_df = pd.concat([payment_df, new_payment], ignore_index=True)
                    payment_df.to_csv(PAYMENT_FILE, index=False)
                    st.success(f"Payment recorded! Receipt: {new_payment['Receipt Number'].iloc[0]}")
        
        with col2:
            st.subheader("Recent Payments")
            if not payment_df.empty:
                recent_payments = payment_df.sort_values('Payment Date', ascending=False).head(10)
                st.dataframe(recent_payments, use_container_width=True)
                
                # Payment statistics
                if 'Status' in payment_df.columns:
                    on_time_rate = (payment_df['Status'] == 'On Time').mean() * 100
                    st.metric("On-Time Payment Rate", f"{on_time_rate:.1f}%")
                
                if 'Late Days' in payment_df.columns:
                    avg_late_days = payment_df['Late Days'].mean()
                    st.metric("Average Late Days", f"{avg_late_days:.1f} days")
            else:
                st.info("No payments recorded yet.")
    
    else:
        st.info("No tenants available. Please add tenants first in the Rent Prediction section.")

def show_analytics_dashboard(history_df, payment_df):
    """Analytics dashboard page."""
    st.header("📈 Analytics Dashboard")
    
    if history_df.empty:
        st.info("No data available for analytics. Add some predictions and payments first.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_tenants = len(history_df)
        st.metric("Total Tenants", total_tenants)
    
    with col2:
        if 'Actual Rent' in history_df.columns:
            avg_rent = history_df['Actual Rent'].mean()
        else:
            avg_rent = history_df['Predicted Rent'].mean()
        st.metric("Average Rent", f"RWF {avg_rent:,.0f}")
    
    with col3:
        if not payment_df.empty and 'Amount' in payment_df.columns:
            total_revenue = payment_df['Amount'].sum()
            st.metric("Total Revenue", f"RWF {total_revenue:,.0f}")
        else:
            st.metric("Total Revenue", "RWF 0")
    
    # Simple tables instead of charts
    col_table1, col_table2 = st.columns(2)
    
    with col_table1:
        st.subheader("Rent by City")
        if 'City' in history_df.columns:
            if 'Actual Rent' in history_df.columns:
                rent_col = 'Actual Rent'
            else:
                rent_col = 'Predicted Rent'
                
            city_rent = history_df.groupby('City')[rent_col].mean().reset_index()
            st.dataframe(city_rent, use_container_width=True)
    
    with col_table2:
        st.subheader("Payment Methods Summary")
        if not payment_df.empty and 'Payment Method' in payment_df.columns:
            payment_summary = payment_df['Payment Method'].value_counts().reset_index()
            payment_summary.columns = ['Payment Method', 'Count']
            st.dataframe(payment_summary, use_container_width=True)
    
    # Payment performance table
    if not payment_df.empty and 'Status' in payment_df.columns:
        st.subheader("Payment Performance")
        payment_status = payment_df['Status'].value_counts().reset_index()
        payment_status.columns = ['Status', 'Count']
        st.dataframe(payment_status, use_container_width=True)

# -------------------------------
if __name__ == "__main__":
    main()
