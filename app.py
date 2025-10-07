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
def create_fresh_model():
    """Create a completely fresh model to avoid compatibility issues."""
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model_v2.joblib")
    
    # Remove old model file if it exists
    old_model_file = os.path.join(os.path.dirname(__file__), "tenant_model.joblib")
    if os.path.exists(old_model_file):
        try:
            os.remove(old_model_file)
            st.info("🔄 Removed incompatible old model file.")
        except:
            pass
    
    st.info("🔄 Creating a new optimized machine learning model...")
    
    # Create realistic sample training data for Rwanda context
    np.random.seed(42)
    n_samples = 1500
    
    # Rwanda-specific city weights (Kigali has higher rents)
    city_weights = {'Kigali': 1.4, 'Musanze': 1.0, 'Huye': 0.9, 'Rubavu': 1.1, 'Nyagatare': 0.8}
    
    sample_data = {
        'BHK': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05]),
        'Size': np.random.randint(300, 2500, n_samples),
        'Bathroom': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.5, 0.1]),
        'Furnishing Status': np.random.choice(['Unfurnished', 'Semi-Furnished', 'Furnished'], n_samples, p=[0.3, 0.5, 0.2]),
        'Tenant Preferred': np.random.choice(['Bachelors', 'Family', 'Bachelors/Family'], n_samples, p=[0.4, 0.4, 0.2]),
        'City': np.random.choice(['Kigali', 'Musanze', 'Huye', 'Rubavu', 'Nyagatare'], n_samples, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
        'Point of Contact': np.random.choice(['Contact Owner', 'Contact Agent', 'Contact Builder'], n_samples, p=[0.5, 0.4, 0.1]),
        'Area Type': np.random.choice(['Super Area', 'Carpet Area', 'Built Area'], n_samples, p=[0.6, 0.3, 0.1])
    }
    
    df = pd.DataFrame(sample_data)
    
    # Encode categorical variables
    le_furnishing = LabelEncoder()
    le_tenant = LabelEncoder()
    le_city = LabelEncoder()
    le_contact = LabelEncoder()
    le_area = LabelEncoder()
    
    df['Furnishing Status_encoded'] = le_furnishing.fit_transform(df['Furnishing Status'])
    df['Tenant Preferred_encoded'] = le_tenant.fit_transform(df['Tenant Preferred'])
    df['City_encoded'] = le_city.fit_transform(df['City'])
    df['Point of Contact_encoded'] = le_contact.fit_transform(df['Point of Contact'])
    df['Area Type_encoded'] = le_area.fit_transform(df['Area Type'])
    
    # Create realistic target variable (rent in RWF) based on Rwanda market
    base_rent = (
        df['BHK'] * 25000 +  # Base per BHK
        df['Size'] * 80 +    # Price per sq.ft
        df['Bathroom'] * 15000 +  # Bathroom premium
        df['Furnishing Status_encoded'] * 20000 +  # Furnishing premium
        df['City_encoded'] * 10000 +  # City premium
        np.random.normal(0, 10000, n_samples)  # Random variation
    )
    
    # Apply Rwanda-specific adjustments
    for i, city in enumerate(df['City']):
        base_rent[i] *= city_weights[city]
    
    # Ensure realistic rent ranges for Rwanda
    base_rent = np.clip(base_rent, 50000, 800000)
    
    # Train model with simpler features to avoid compatibility issues
    feature_columns = [
        'BHK', 'Size', 'Bathroom', 
        'Furnishing Status_encoded', 'Tenant Preferred_encoded',
        'City_encoded', 'Point of Contact_encoded', 'Area Type_encoded'
    ]
    
    X = df[feature_columns]
    y = base_rent
    
    model = RandomForestRegressor(
        n_estimators=50,  # Smaller for compatibility
        max_depth=10,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X, y)
    
    # Save model and encoders in a simple format
    model_data = {
        'model': model,
        'encoders': {
            'furnishing': le_furnishing,
            'tenant': le_tenant,
            'city': le_city,
            'contact': le_contact,
            'area': le_area
        },
        'feature_columns': feature_columns,
        'model_version': 'v2_simple',
        'created_date': str(datetime.now())
    }
    
    joblib.dump(model_data, MODEL_FILE)
    st.success("✅ New model created successfully!")
    return model_data

def load_model():
    """Load the ML model or create a fresh one."""
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model_v2.joblib")
    
    if not os.path.exists(MODEL_FILE):
        return create_fresh_model()
    
    try:
        model_data = joblib.load(MODEL_FILE)
        # Verify the model has all required components
        required_keys = ['model', 'encoders', 'feature_columns']
        if all(key in model_data for key in required_keys):
            return model_data
        else:
            st.warning("Model file exists but is incomplete. Creating new model...")
            return create_fresh_model()
    except Exception as e:
        st.warning(f"Could not load existing model: {e}. Creating a new optimized model...")
        return create_fresh_model()

def load_history():
    """Load existing history CSV or create empty DataFrame."""
    HISTORY_FILE = os.path.join(os.path.dirname(__file__), "tenant_history.csv")
    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            # Ensure all required columns exist
            required_columns = [
                'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
                'City','Point of Contact','Area Locality','Posted On','Area Type',
                'Floor','Predicted Rent', 'Actual Rent', 'Tenant Name', 'Phone Number',
                'Contract Start', 'Contract End', 'Payment Method', 'Security Deposit'
            ]
            for col in required_columns:
                if col not in history_df.columns:
                    history_df[col] = None
            return history_df, HISTORY_FILE
        except Exception as e:
            st.warning(f"Error loading history file: {e}. Creating new history...")
    else:
        st.info("No existing history found. Starting with fresh records.")
    
    # Create new history dataframe
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
        try:
            payment_df = pd.read_csv(PAYMENT_FILE)
            
            # Ensure all required columns exist
            required_columns = [
                'Tenant Name', 'Phone Number', 'Payment Date', 'Due Date', 
                'Amount', 'Payment Method', 'Status', 'Late Days', 'Receipt Number'
            ]
            for col in required_columns:
                if col not in payment_df.columns:
                    payment_df[col] = None
            
            # Safely convert date columns
            if 'Payment Date' in payment_df.columns and len(payment_df) > 0:
                payment_df['Payment Date'] = pd.to_datetime(payment_df['Payment Date'], errors='coerce')
            if 'Due Date' in payment_df.columns and len(payment_df) > 0:
                payment_df['Due Date'] = pd.to_datetime(payment_df['Due Date'], errors='coerce')
                
            return payment_df, PAYMENT_FILE
        except Exception as e:
            st.warning(f"Error loading payment file: {e}. Creating new payment history...")
    
    # Create new payment dataframe
    payment_df = pd.DataFrame(columns=[
        'Tenant Name', 'Phone Number', 'Payment Date', 'Due Date', 
        'Amount', 'Payment Method', 'Status', 'Late Days', 'Receipt Number'
    ])
    return payment_df, PAYMENT_FILE

def preprocess_input(input_data, encoders):
    """Preprocess input data for prediction with error handling."""
    try:
        # Create encoded versions
        furnishing_encoded = encoders['furnishing'].transform([input_data['Furnishing Status'].iloc[0]])[0]
        tenant_encoded = encoders['tenant'].transform([input_data['Tenant Preferred'].iloc[0]])[0]
        city_encoded = encoders['city'].transform([input_data['City'].iloc[0]])[0]
        contact_encoded = encoders['contact'].transform([input_data['Point of Contact'].iloc[0]])[0]
        area_encoded = encoders['area'].transform([input_data['Area Type'].iloc[0]])[0]
        
        # Create feature array in correct order
        features = np.array([[
            input_data['BHK'].iloc[0],
            input_data['Size'].iloc[0],
            input_data['Bathroom'].iloc[0],
            furnishing_encoded,
            tenant_encoded,
            city_encoded,
            contact_encoded,
            area_encoded
        ]])
        
        return features
        
    except Exception as e:
        st.error(f"Error preprocessing input: {str(e)}")
        # Return default features
        return np.array([[2, 950, 2, 1, 1, 0, 0, 0]])

# -------------------------------
def main():
    # Page config
    st.set_page_config(
        page_title="Rent Prediction & Payment System", 
        page_icon="🏠", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Show loading message
    with st.spinner("Loading Tenant Management System..."):
        # Load model and history
        try:
            model_data = load_model()
            model = model_data['model']
            encoders = model_data['encoders']
            
            history_df, HISTORY_FILE = load_history()
            payment_df, PAYMENT_FILE = load_payment_history()
        except Exception as e:
            st.error(f"Error initializing application: {str(e)}")
            st.info("Please refresh the page to try again.")
            st.stop()

    # App title
    st.markdown(
        """
        <h1 style="text-align:center; color:#2E86C1;">🏡 Rwanda Tenant Rent Prediction System</h1>
        <p style="text-align:center; color:#7D3C98; font-size:18px;">
        Smart rent prediction and payment tracking for Rwandan landlords
        </p>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Go to", 
                           ["Rent Prediction", "Tenant Management", "Payment Tracking", "Analytics Dashboard"])
    
    # System info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Info")
    st.sidebar.info(f"Tenants: {len(history_df)} | Payments: {len(payment_df)}")
    
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
    st.markdown("Predict monthly rental prices based on property features and location.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Property Details")
        BHK = st.number_input("BHK", min_value=1, max_value=10, value=2, key="pred_bhk")
        Size = st.number_input("Size (sq.ft)", min_value=100, max_value=10000, value=950, key="pred_size")
        Bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, key="pred_bath")
        Furnishing_Status = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"], key="pred_furnish")
        Area_Type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"], key="pred_area_type")
        Floor = st.text_input("Floor (e.g. '5 out of 10')", "5 out of 10", key="pred_floor")
    
    with col2:
        st.subheader("👥 Tenant & Location")
        Tenant_Preferred = st.selectbox("Tenant Preferred", ["Bachelors", "Family", "Bachelors/Family"], key="pred_tenant")
        City = st.selectbox("City", ["Kigali", "Musanze", "Huye", "Rubavu", "Nyagatare"], key="pred_city")
        Point_of_Contact = st.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"], key="pred_contact")
        Area_Locality = st.text_input("Area Locality", "Nyarugenge", key="pred_locality")
        Posted_On = st.date_input("Posted On", value=date.today(), key="pred_posted")
        
        # Additional tenant info for management
        st.subheader("📝 Tenant Details (Optional)")
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
    if st.button("🎯 Predict Rent", use_container_width=True, type="primary"):
        with st.spinner("Calculating optimal rent..."):
            try:
                # Preprocess and predict
                processed_input = preprocess_input(input_data, encoders)
                predicted_rent = model.predict(processed_input)[0]
                
                # Ensure rent is reasonable for Rwanda context
                predicted_rent = max(50000, min(800000, predicted_rent))

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
                    <div style="background-color:#D6EAF8; padding:20px; border-radius:12px; border-left: 5px solid #2E86C1;">
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
                    if predicted_rent > 200000:
                        st.error("""
                        **💰 High Rent Alert:**
                        - Consider flexible payment plans
                        - Require 2-3 months security deposit
                        - Verify tenant income stability
                        - Common in Kigali premium areas
                        """)
                    elif predicted_rent > 100000:
                        st.warning("""
                        **🏠 Moderate Rent Range:**
                        - Standard 2-month deposit recommended
                        - Good for employed professionals
                        - Popular in urban centers
                        """)
                    else:
                        st.success("""
                        **💸 Affordable Range:**
                        - 1-2 months security deposit
                        - High tenant demand expected
                        - Good for starter apartments
                        """)
                
                with col_advice2:
                    st.info("""
                    **📱 Payment Recommendations for Rwanda:**
                    - ✅ **Mobile Money** (Momo, Airtel Money)
                    - ✅ **Bank Transfer** (for formal agreements)
                    - ✅ **Cash** (with signed receipt)
                    - ❌ Informal cash payments without receipt
                    
                    **📍 Market Insight:**
                    Kigali: RWF 80,000 - 500,000+
                    Other cities: RWF 50,000 - 200,000
                    """)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check your input values and try again. Using default prediction model.")

# ... (Keep the rest of the functions exactly the same as in the previous working version)
# [Include show_tenant_management, show_payment_tracking, show_analytics_dashboard functions here]
# They remain unchanged from the previous working version

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
        
        if available_columns:
            st.dataframe(display_df[available_columns], use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
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
                    try:
                        tenant_name = history_df.iloc[delete_index].get('Tenant Name', 'Unknown')
                        history_df.drop(index=delete_index, inplace=True)
                        history_df.reset_index(drop=True, inplace=True)
                        history_df.to_csv(HISTORY_FILE, index=False)
                        st.success(f"Record for {tenant_name} deleted successfully.")
                    except Exception as e:
                        st.error(f"Error deleting record: {str(e)}")
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
                    try:
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
                    except Exception as e:
                        st.error(f"Error recording payment: {str(e)}")
        
        with col2:
            st.subheader("Recent Payments")
            if not payment_df.empty:
                try:
                    # Sort by Payment Date if it exists and has valid data
                    if 'Payment Date' in payment_df.columns and not payment_df['Payment Date'].isna().all():
                        recent_payments = payment_df.sort_values('Payment Date', ascending=False).head(10)
                    else:
                        recent_payments = payment_df.head(10)
                    st.dataframe(recent_payments, use_container_width=True)
                    
                    # Payment statistics
                    if 'Status' in payment_df.columns and not payment_df['Status'].isna().all():
                        on_time_rate = (payment_df['Status'] == 'On Time').mean() * 100
                        st.metric("On-Time Payment Rate", f"{on_time_rate:.1f}%")
                    
                    if 'Late Days' in payment_df.columns and not payment_df['Late Days'].isna().all():
                        avg_late_days = payment_df['Late Days'].mean()
                        st.metric("Average Late Days", f"{avg_late_days:.1f} days")
                except Exception as e:
                    st.error(f"Error displaying payment data: {str(e)}")
                    st.dataframe(payment_df, use_container_width=True)
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
        try:
            if 'Actual Rent' in history_df.columns:
                avg_rent = history_df['Actual Rent'].mean()
            else:
                avg_rent = history_df['Predicted Rent'].mean()
            st.metric("Average Rent", f"RWF {avg_rent:,.0f}")
        except:
            st.metric("Average Rent", "RWF 0")
    
    with col3:
        try:
            if not payment_df.empty and 'Amount' in payment_df.columns:
                total_revenue = payment_df['Amount'].sum()
                st.metric("Total Revenue", f"RWF {total_revenue:,.0f}")
            else:
                st.metric("Total Revenue", "RWF 0")
        except:
            st.metric("Total Revenue", "RWF 0")
    
    # Simple tables instead of charts
    col_table1, col_table2 = st.columns(2)
    
    with col_table1:
        st.subheader("Rent by City")
        if 'City' in history_df.columns:
            try:
                if 'Actual Rent' in history_df.columns:
                    rent_col = 'Actual Rent'
                else:
                    rent_col = 'Predicted Rent'
                    
                city_rent = history_df.groupby('City')[rent_col].mean().reset_index()
                st.dataframe(city_rent, use_container_width=True)
            except Exception as e:
                st.info("Unable to display rent by city data")
    
    with col_table2:
        st.subheader("Payment Methods Summary")
        if not payment_df.empty and 'Payment Method' in payment_df.columns:
            try:
                payment_summary = payment_df['Payment Method'].value_counts().reset_index()
                payment_summary.columns = ['Payment Method', 'Count']
                st.dataframe(payment_summary, use_container_width=True)
            except Exception as e:
                st.info("Unable to display payment methods summary")
        else:
            st.info("No payment data available")
    
    # Payment performance table
    if not payment_df.empty and 'Status' in payment_df.columns:
        st.subheader("Payment Performance")
        try:
            payment_status = payment_df['Status'].value_counts().reset_index()
            payment_status.columns = ['Status', 'Count']
            st.dataframe(payment_status, use_container_width=True)
        except Exception as e:
            st.info("Unable to display payment performance data")

# -------------------------------
if __name__ == "__main__":
    main()
