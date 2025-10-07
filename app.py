# -*- coding: utf-8 -*-
"""
Tenant Rent Prediction & Payment Tracking System
Author: GODSON
"""

import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
def load_model():
    """Load the ML model (joblib only)."""
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model.joblib")
    if not os.path.exists(MODEL_FILE):
        st.error("Model file not found. Please upload tenant_model.joblib.")
        st.stop()
    return joblib.load(MODEL_FILE)

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
        payment_df['Payment Date'] = pd.to_datetime(payment_df['Payment Date'])
        payment_df['Due Date'] = pd.to_datetime(payment_df['Due Date'])
    else:
        payment_df = pd.DataFrame(columns=[
            'Tenant Name', 'Phone Number', 'Payment Date', 'Due Date', 
            'Amount', 'Payment Method', 'Status', 'Late Days', 'Receipt Number'
        ])
    return payment_df, PAYMENT_FILE

# -------------------------------
def main():
    # Page config
    st.set_page_config(page_title="Rent Prediction & Payment System", page_icon="🏠", layout="wide")

    # Load model and history
    model = load_model()
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
        show_rent_prediction(model, history_df, HISTORY_FILE)
    elif page == "Tenant Management":
        show_tenant_management(history_df, HISTORY_FILE)
    elif page == "Payment Tracking":
        show_payment_tracking(history_df, payment_df, PAYMENT_FILE)
    elif page == "Analytics Dashboard":
        show_analytics_dashboard(history_df, payment_df)

def show_rent_prediction(model, history_df, HISTORY_FILE):
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
        Area_Locality = st.text_input("Area Locality", key="pred_locality")
        Posted_On = st.date_input("Posted On", key="pred_posted")
        
        # Additional tenant info for management
        st.subheader("Tenant Details (Optional)")
        tenant_name = st.text_input("Tenant Name", key="pred_name")
        phone_number = st.text_input("Phone Number", key="pred_phone")
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
        predicted_rent = model.predict(input_data)[0]

        # Save to history with additional tenant info
        input_data['Predicted Rent'] = predicted_rent
        input_data['Actual Rent'] = actual_rent if actual_rent > 0 else predicted_rent
        input_data['Tenant Name'] = tenant_name
        input_data['Phone Number'] = phone_number
        input_data['Contract Start'] = str(date.today())
        input_data['Contract End'] = str(date.today().replace(year=date.today().year + 1))
        input_data['Payment Method'] = "Mobile Money"  # Default for Rwanda
        input_data['Security Deposit'] = (actual_rent if actual_rent > 0 else predicted_rent) * 2  # 2 months deposit
        
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
            if predicted_rent > 90000:
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

def show_tenant_management(history_df, HISTORY_FILE):
    """Tenant management page."""
    st.header("👥 Tenant Management")
    
    if not history_df.empty:
        # Display current tenants
        st.subheader("Current Tenants")
        display_df = history_df[['Tenant Name', 'Phone Number', 'City', 'Area Locality', 
                               'Predicted Rent', 'Actual Rent', 'Contract Start', 'Contract End']].copy()
        display_df['Actual Rent'] = display_df['Actual Rent'].fillna(display_df['Predicted Rent'])
        st.dataframe(display_df, use_container_width=True)
        
        # Tenant actions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Update Tenant Info")
            tenant_index = st.selectbox("Select Tenant", range(len(history_df)), 
                                      format_func=lambda x: f"{history_df.iloc[x]['Tenant Name']} - {history_df.iloc[x]['Phone Number']}")
            
            if st.button("Update Payment Method"):
                new_method = st.selectbox("Payment Method", 
                                        ["Mobile Money", "Bank Transfer", "Cash", "Other"])
                history_df.at[tenant_index, 'Payment Method'] = new_method
                history_df.to_csv(HISTORY_FILE, index=False)
                st.success(f"Updated payment method for {history_df.iloc[tenant_index]['Tenant Name']}")
        
        with col2:
            st.subheader("Delete Tenant Record")
            delete_index = st.number_input("Row index to delete", min_value=0, 
                                         max_value=len(history_df)-1, step=1, key="delete_idx")
            if st.button("🗑️ Delete Tenant Record", type="secondary"):
                tenant_name = history_df.iloc[delete_index]['Tenant Name']
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
            tenant_index = st.selectbox("Select Tenant", range(len(history_df)),
                                      format_func=lambda x: f"{history_df.iloc[x]['Tenant Name']} - RWF {history_df.iloc[x]['Actual Rent']:,.0f}",
                                      key="payment_tenant")
            
            tenant_data = history_df.iloc[tenant_index]
            due_date = st.date_input("Due Date", value=date.today().replace(day=1))
            payment_date = st.date_input("Payment Date", value=date.today())
            amount = st.number_input("Amount Paid", value=float(tenant_data['Actual Rent']))
            payment_method = st.selectbox("Payment Method", 
                                        ["Mobile Money", "Bank Transfer", "Cash", "Other"])
            
            # Calculate late days
            late_days = max(0, (payment_date - due_date).days)
            status = "On Time" if late_days == 0 else "Late"
            
            if st.button("💾 Record Payment", use_container_width=True):
                new_payment = pd.DataFrame({
                    'Tenant Name': [tenant_data['Tenant Name']],
                    'Phone Number': [tenant_data['Phone Number']],
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
                on_time_rate = (payment_df['Status'] == 'On Time').mean() * 100
                avg_late_days = payment_df['Late Days'].mean()
                
                st.metric("On-Time Payment Rate", f"{on_time_rate:.1f}%")
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
        avg_rent = history_df['Actual Rent'].mean()
        st.metric("Average Rent", f"RWF {avg_rent:,.0f}")
    
    with col3:
        if not payment_df.empty:
            total_revenue = payment_df['Amount'].sum()
            st.metric("Total Revenue", f"RWF {total_revenue:,.0f}")
        else:
            st.metric("Total Revenue", "RWF 0")
    
    # Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Rent Distribution by City")
        if 'City' in history_df.columns:
            city_rent = history_df.groupby('City')['Actual Rent'].mean().reset_index()
            fig = px.bar(city_rent, x='City', y='Actual Rent', 
                        title="Average Rent by City")
            st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.subheader("Payment Methods")
        if not payment_df.empty:
            payment_methods = payment_df['Payment Method'].value_counts()
            fig = px.pie(values=payment_methods.values, names=payment_methods.index,
                        title="Payment Method Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Payment performance
    if not payment_df.empty:
        st.subheader("Payment Performance")
        payment_status = payment_df['Status'].value_counts()
        fig = px.pie(values=payment_status.values, names=payment_status.index,
                    title="Payment Status Distribution")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
if __name__ == "__main__":
    main()