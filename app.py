import streamlit as st
import pandas as pd
import joblib

# Load both the trained model AND the preprocessor
try:
    model = joblib.load("outputs/models/best_model.pkl")
    preprocessor = joblib.load("outputs/models/preprocessor.pkl")
    st.success("✅ Model and preprocessor loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

st.title("🚗 CO₂ Emissions Prediction App")
st.write("Enter complete vehicle details to predict CO₂ emissions")

# Create input form with all required fields
with st.form("prediction_form"):
    st.subheader("Vehicle Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric inputs
        engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=10.0, step=0.1, value=2.0)
        cylinders = st.number_input("Cylinders", min_value=2, max_value=16, step=1, value=4)
        fuel_consumption_city = st.number_input("Fuel Consumption City (L/100 km)", min_value=1.0, max_value=30.0, step=0.1, value=9.0)
        fuel_consumption_hwy = st.number_input("Fuel Consumption Hwy (L/100 km)", min_value=1.0, max_value=30.0, step=0.1, value=7.0)
        fuel_consumption_comb = st.number_input("Fuel Consumption Comb (L/100 km)", min_value=1.0, max_value=30.0, step=0.1, value=8.0)
    
    with col2:
        # Categorical inputs (you'll need to adjust these options based on your actual data)
        make = st.selectbox("Make", [
            "ACURA", "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BENTLEY", "BMW", "BUICK", 
            "CADILLAC", "CHEVROLET", "CHRYSLER", "DODGE", "FERRARI", "FIAT", "FORD", 
            "GENESIS", "GMC", "HONDA", "HYUNDAI", "INFINITI", "JAGUAR", "JEEP", "KIA", 
            "LAMBORGHINI", "LAND ROVER", "LEXUS", "LINCOLN", "MASERATI", "MAZDA", 
            "MERCEDES-BENZ", "MINI", "MITSUBISHI", "NISSAN", "PORSCHE", "RAM", 
            "ROLLS-ROYCE", "SCION", "SMART", "SUBARU", "TOYOTA", "VOLKSWAGEN", "VOLVO"
        ])
        
        model_name = st.text_input("Model", value="Civic", help="Enter the vehicle model name")
        
        vehicle_class = st.selectbox("Vehicle Class", [
            "COMPACT", "MID-SIZE", "FULL-SIZE", "SUV - SMALL", "SUV - STANDARD", 
            "PICKUP TRUCK - SMALL", "PICKUP TRUCK - STANDARD", "MINIVAN", 
            "STATION WAGON - SMALL", "STATION WAGON - MID-SIZE", "SUBCOMPACT", 
            "TWO-SEATER", "SPECIAL PURPOSE VEHICLE"
        ])
        
        fuel_type = st.selectbox("Fuel Type", [
            "X", "Z", "D", "E"  # You may need to adjust these based on your data
        ], help="X=Regular gasoline, Z=Premium gasoline, D=Diesel, E=Ethanol")
        
        transmission = st.selectbox("Transmission", [
            "A4", "A5", "A6", "A7", "A8", "A9", "A10", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10",
            "AV", "AV6", "AV7", "AV8", "AV10", "CVT", "M5", "M6", "M7"
        ], help="A=Automatic, M=Manual, AV=Continuously variable, CVT=Continuously variable")
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict CO₂ Emission", type="primary")

if submitted:
    try:
        # Create DataFrame with ALL required columns in the correct order
        input_data = pd.DataFrame([[
            make, model_name, vehicle_class, engine_size, cylinders, transmission, 
            fuel_type, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb
        ]], columns=[
            "Make", "Model", "Vehicle Class", "Engine Size(L)", "Cylinders", 
            "Transmission", "Fuel Type", "Fuel Consumption City (L/100 km)", 
            "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)"
        ])
        
        # Apply the same preprocessing as during training
        input_data_processed = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_processed)[0]
        
        # Display result with styling
        st.success(f"🎯 **Estimated CO₂ Emission: {prediction:.1f} g/km**")
        
        # Add emission category
        if prediction < 150:
            st.info("✅ **Low emissions** - Environmentally friendly vehicle!")
        elif prediction < 200:
            st.warning("⚠️ **Moderate emissions** - Average environmental impact")
        elif prediction < 300:
            st.error("🚨 **High emissions** - Consider a more efficient vehicle")
        else:
            st.error("💀 **Very high emissions** - Significant environmental impact")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.write("**Input data shape:**", input_data.shape if 'input_data' in locals() else "Not created")
            st.write("**Input data columns:**", input_data.columns.tolist() if 'input_data' in locals() else "Not created")
            st.write("**Error details:**", str(e))

# Information sections
st.write("---")

col1, col2 = st.columns(2)

with col1:
    with st.expander("ℹ️ Model Performance"):
        st.write("**Model Type:** Random Forest Regressor")
        st.write("**Performance Metrics:**")
        st.write("- R² Score: 0.9965")
        st.write("- RMSE: 3.48 g/km")
        st.write("- MAE: 1.83 g/km")

with col2:
    with st.expander("🔧 Fuel Type Guide"):
        st.write("**Fuel Type Codes:**")
        st.write("- **X**: Regular gasoline")
        st.write("- **Z**: Premium gasoline") 
        st.write("- **D**: Diesel")
        st.write("- **E**: Ethanol (E85)")
        
with st.expander("🚗 Transmission Guide"):
    st.write("**Transmission Codes:**")
    st.write("- **A + number**: Automatic (A4 = 4-speed auto)")
    st.write("- **M + number**: Manual (M6 = 6-speed manual)")
    st.write("- **AV**: Continuously Variable Automatic")
    st.write("- **CVT**: Continuously Variable Transmission")