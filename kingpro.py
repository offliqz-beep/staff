# -*- coding: utf-8 -*-
"""
Tenant Rent Prediction Web App (Cloud-ready)
Author: GODSON
"""

import streamlit as st
import pandas as pd
import os
import sys

# -------------------------------
def load_model():
    """Load the ML model with simple error handling (no long warnings)."""
    MODEL_FILE = os.path.join(os.path.dirname(__file__), "tenant_model.joblib")
    
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ Model file not found at: {MODEL_FILE}")
        st.stop()
    
    try:
        import joblib
        model = joblib.load(MODEL_FILE)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load saved model, switching to demo mode.\nError: {str(e)}")
        return None

def load_history():
    """Load existing history CSV or create empty DataFrame."""
    HISTORY_FILE = os.path.join(os.path.dirname(__file__), "tenant_history.csv")
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        st.info(f"📋 Loaded {len(history_df)} historical records")
    else:
        history_df = pd.DataFrame(columns=[
            'BHK','Size','Bathroom','Furnishing Status','Tenant Preferred',
            'City','Point of Contact','Area Locality','Posted On','Area Type','Floor','Predicted Rent'
        ])
        st.info("📋 Created new history file")
    return history_df, HISTORY_FILE

# -------------------------------
def create_demo_model():
    """Create a simple demo model if the real one fails to load."""
    st.warning("🔄 Using demo model for testing purposes...")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 11)
        y_dummy = np.random.uniform(5000, 50000, 100)
        model.fit(X_dummy, y_dummy)
        
        st.success("✅ Demo model ready!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to create demo model: {e}")
        return None

def main():
    # Page config
    st.set_page_config(page_title="Tenant Rent Prediction", page_icon="🏠", layout="wide")
    
    # App title
    st.markdown(
        """
        <h1 style="text-align:center; color:#2E86C1;">🏡 Tenant Rent Prediction System</h1>
        <p style="text-align:center; color:#7D3C98; font-size:18px;">
        Enter tenant & property details to predict monthly rent and manage records.
        </p>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Environment info
    with st.expander("🔍 Environment Information"):
        st.write(f"Python version: {sys.version}")
        try:
            import sklearn
            st.write(f"scikit-learn version: {sklearn.__version__}")
        except:
            st.write("❌ scikit-learn: Not installed")
        try:
            import joblib
            st.write(f"joblib version: {joblib.__version__}")
        except:
            st.write("❌ joblib: Not installed")
    
    # Load model with fallback
    model = load_model()
    if model is None:
        model = create_demo_model()
    
    if model is None:
        st.error("❌ Cannot proceed without a model. Please install dependencies and try again.")
        st.code("pip install scikit-learn joblib pandas streamlit")
        return
    
    # Load history
    history_df, HISTORY_FILE = load_history()

    # Sidebar inputs
    st.sidebar.header("🔧 Tenant & Property Details")
    BHK = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=2)
    Size = st.sidebar.number_input("Size (sq.ft)", min_value=100, max_value=10000, value=950)
    Bathroom = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    Furnishing_Status = st.sidebar.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
    Tenant_Preferred = st.sidebar.selectbox("Tenant Preferred", ["Bachelors", "Family", "Bachelors/Family"])
    City = st.sidebar.text_input("City", "Mumbai")
    Point_of_Contact = st.sidebar.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])
    Area_Locality = st.sidebar.text_input("Area Locality", "Bandra West")
    Posted_On = st.sidebar.date_input("Posted On")
    Area_Type = st.sidebar.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
    Floor = st.sidebar.text_input("Floor (e.g. '5 out of 10')", "5 out of 10")

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
    if st.sidebar.button("Predict Rent"):
        try:
            if hasattr(model, 'feature_names_in_'):
                predicted_rent = model.predict(input_data)[0]
            else:
                base_rent = (BHK * 10000 + Size * 50 + Bathroom * 5000)
                predicted_rent = max(5000, min(200000, base_rent))
                st.info("🔸 Using demo prediction")
            
            input_data['Predicted Rent'] = predicted_rent
            history_df = pd.concat([history_df, input_data], ignore_index=True)
            history_df.to_csv(HISTORY_FILE, index=False)

            st.markdown(
                f"""
                <div style="background-color:#D6EAF8; padding:20px; border-radius:12px;">
                    <h2 style="color:#154360;"> Predicted Monthly Rent: ₹{predicted_rent:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True
            )

            if predicted_rent > 90000:
                st.error("💰 High rent: tenant may struggle to pay on time.")
            else:
                st.success("✅ Rent is reasonable: tenant likely to pay on time.")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

    # Old predictions table with delete
    st.subheader("📊 Prediction History")
    if not history_df.empty:
        st.dataframe(history_df)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Delete a row by index:**")
            delete_index = st.number_input("Row index to delete", min_value=0, max_value=len(history_df)-1, step=1)
        with col2:
            if st.button("🗑️ Delete Row"):
                history_df.drop(index=delete_index, inplace=True)
                history_df.reset_index(drop=True, inplace=True)
                history_df.to_csv(HISTORY_FILE, index=False)
                st.success(f"✅ Row {delete_index} deleted successfully.")
                st.rerun()
    else:
        st.info("No prediction history yet. Make your first prediction!")

# -------------------------------
if __name__ == "__main__":
    main()

