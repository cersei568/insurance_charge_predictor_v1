import streamlit as st
import pandas as pd # type: ignore
import numpy as np # type: ignore

# Try to import PyCaret with error handling
try:
    from pycaret.datasets import get_data
    from pycaret.regression import load_model, predict_model
    PYCARET_AVAILABLE = True
except ImportError as e:
    st.error(f"PyCaret import error: {e}")
    PYCARET_AVAILABLE = False

# Try to import SHAP with error handling
try:
    import shap # type: ignore
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="💰 Insurance Charge Predictor", 
    page_icon="💰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-color: #3b82f6;
        --primary-dark: #2563eb;
        --secondary-color: #10b981;
        --accent-color: #8b5cf6;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        --card-background: rgba(255, 255, 255, 0.95);
        --glass-background: rgba(255, 255, 255, 0.05);
        --text-primary: #1e293b;
        --text-secondary: rgba(30, 41, 59, 0.8);
        --text-muted: rgba(30, 41, 59, 0.6);
        --text-light: rgba(255, 255, 255, 0.9);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        --border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: var(--background-gradient);
        min-height: 100vh;
    }
    
    /* Hide default streamlit elements but keep sidebar controls */
    .css-18e3th9 {
        padding-top: 2rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Keep sidebar toggle button visible */
    .css-1rs6os, .css-1vbkxwb {
        visibility: visible !important;
        z-index: 999999 !important;
        position: relative !important;
    }
    
    /* Ensure rerun button is visible */
    .stButton[data-testid="stBaseButton-secondary"] {
        visibility: visible !important;
        position: relative !important;
        z-index: 999999 !important;
    }
    
    /* Make sure sidebar toggle is always accessible */
    button[aria-label="Close sidebar"] {
        visibility: visible !important;
        z-index: 999999 !important;
    }
    
    button[aria-label="Open sidebar"] {
        visibility: visible !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
    }
    
    /* Custom header */
    .app-header {
        background: var(--glass-background);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-xl);
        animation: slideDown 0.8s ease-out;
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: var(--text-light);
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--glass-background);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] * {
        color: var(--text-light) !important;
    }

    /* Target the actual scrollable sidebar container */
[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    max-height: 100vh;
    scrollbar-width: thin;
    scrollbar-color: #3b82f6 rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
    width: 8px;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #3b82f6, #10b981);
    border-radius: 10px;
}

[data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #2563eb, #10b981);
}


    /* Sidebar text */
    .css-1d391kg * {
        color: var(--text-light) !important;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: var(--text-light) !important;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-color));
    }
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stRadio > div > div > div {
        background: var(--glass-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px;
        color: var(--text-light) !important;
        backdrop-filter: blur(10px);
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Input labels */
    .stNumberInput label,
    .stSelectbox label,
    .stRadio label {
        color: var(--text-light) !important;
        font-weight: 500;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Radio buttons */
    .stRadio > div > div > div > label {
        color: var(--text-light) !important;
    }
    
    /* Selectbox options */
    .stSelectbox > div > div > div > div {
        background: var(--card-background) !important;
        color: var(--text-primary) !important;
    }
    
    /* Main content cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-xl);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 32px 64px -12px rgba(0, 0, 0, 0.4);
    }
    
    .prediction-card * {
        color: var(--text-primary) !important;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, var(--secondary-color), #059669);
        color: white !important;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: var(--shadow-xl);
        animation: pulse 2s infinite;
    }
    
    .prediction-result h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    
    .prediction-result h2 {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.5rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: var(--card-background);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border-color);
    }
    
    .stDataFrame table {
        color: var(--text-primary) !important;
    }
    
    .stDataFrame th {
        background: var(--primary-color) !important;
        color: white !important;
        font-weight: 600;
    }
    
    .stDataFrame td {
        color: var(--text-primary) !important;
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-xl);
        transition: all 0.3s ease;
    }
    
    .glass-card * {
        color: var(--text-primary) !important;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 32px 64px -12px rgba(0, 0, 0, 0.6);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Suggestions styling */
    .suggestion-item {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .suggestion-item:hover {
        background: rgba(59, 130, 246, 0.15);
        transform: translateX(5px);
    }
    
    /* Section headers */
    .section-header {
        color: var(--text-primary) !important;
        font-weight: 600;
        font-size: 1.5rem;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        border-radius: 2px;
    }
    
    /* Sidebar section headers */
    .css-1d391kg .section-header {
        color: var(--text-light) !important;
    }
    
    /* BMI display */
    .bmi-display {
        background: linear-gradient(135deg, var(--accent-color), #7c3aed);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: var(--shadow-lg);
    }
    
    .bmi-display h3 {
        color: white !important;
        margin: 0;
        font-size: 1.2rem;
    }
    
    /* Animations */
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--primary-dark), var(--secondary-color));
    }
    
    /* CSS to ensure sidebar toggle is always visible */
    .stApp > div:first-child {
        position: relative;
    }
    
    /* Make sure the hamburger menu is always accessible */
    .css-1rs6os .css-1vbkxwb {
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        z-index: 999999 !important;
        background: var(--glass-background) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        color: var(--text-light) !important;
    }
    
    /* Ensure the close button in sidebar is styled */
    .css-1d391kg button[aria-label="Close sidebar"] {
        background: var(--glass-background) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-light) !important;
    }
    
    /* Style the main container to account for fixed elements */
    .block-container {
        padding-top: 4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- CONSTANTS --------------------
MODEL_NAME = 'best_final_model'

# -------------------- CACHED FUNCTIONS --------------------
@st.cache_data
def get_model():
    if not PYCARET_AVAILABLE:
        return None
    try:
        return load_model(MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def get_all_participants():
    if not PYCARET_AVAILABLE:
        # Return sample data if PyCaret is not available
        return create_sample_data()
    
    try:
        model = get_model()
        if model is None:
            return create_sample_data()
        
        all_df = get_data('insurance', verbose=False)
        df_with_clusters = predict_model(model, data=all_df)
        return df_with_clusters
    except Exception as e:
        st.error(f"Error getting data: {e}")
        return create_sample_data()

def create_sample_data():
    """Create sample insurance data for demo purposes"""
    np.random.seed(42)
    n_samples = 100
    
    ages = np.random.randint(18, 65, n_samples)
    sexes = np.random.choice(['male', 'female'], n_samples)
    bmis = np.random.normal(26, 4, n_samples).clip(18, 40)
    children = np.random.choice([0, 1, 2, 3, 4, 5], n_samples)
    smokers = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    regions = np.random.choice(['southeast', 'southwest', 'northeast', 'northwest'], n_samples)
    
    # Simple charge calculation for demo
    base_charges = ages * 100 + bmis * 200 + children * 500
    smoking_multiplier = np.where(np.array(smokers) == 'yes', 2.5, 1.0)
    charges = base_charges * smoking_multiplier + np.random.normal(0, 1000, n_samples)
    charges = np.maximum(charges, 1000)  # Minimum charge
    
    return pd.DataFrame({
        'age': ages,
        'sex': sexes,
        'bmi': bmis.round(1),
        'children': children,
        'smoker': smokers,
        'region': regions,
        'charges': charges.round(2)
    })

def predict_insurance_charge(person_data):
    """Simple prediction function if PyCaret is not available"""
    age = person_data['age'].iloc[0]
    sex = person_data['sex'].iloc[0]
    bmi = person_data['bmi'].iloc[0]
    children = int(person_data['children'].iloc[0])
    smoker = person_data['smoker'].iloc[0]
    region = person_data['region'].iloc[0]
    
    # Simple prediction formula for demo
    base_charge = age * 150 + bmi * 300 + children * 700
    
    # Gender factor
    if sex == 'male':
        base_charge *= 1.05
    
    # Smoking factor
    if smoker == 'yes':
        base_charge *= 2.8
    
    # Region factors
    region_factors = {
        'southeast': 1.1,
        'southwest': 0.9,
        'northeast': 1.05,
        'northwest': 0.95
    }
    base_charge *= region_factors.get(region, 1.0)
    
    # Add some randomness
    base_charge += np.random.normal(0, 500)
    
    return max(base_charge, 1000)  # Minimum charge

# -------------------- BEAUTIFUL HEADER --------------------
st.markdown("""
    <div class="app-header">
        <h1 class="app-title">💰 Insurance Charge Predictor</h1>
        <p class="app-subtitle">AI-powered insurance cost estimation with personalized recommendations</p>
    </div>
""", unsafe_allow_html=True)

# Add floating rerun button
col1, col2, col3 = st.columns([1, 2, 1])
with col3:
    if st.button("🔄 Recalculate", key="rerun_button", help="Recalculate with current values"):
        st.rerun()

# Show status of dependencies
if not PYCARET_AVAILABLE:
    st.warning("⚠️ PyCaret not available. Running in demo mode with simplified predictions.")

if not SHAP_AVAILABLE:
    st.info("ℹ️ SHAP not available. Advanced explanations will be simplified.")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown('<h2 class="section-header">👤 Personal Information</h2>', unsafe_allow_html=True)
    st.markdown("**Tell us about yourself and we'll calculate your insurance charge**")
    
    # Personal details
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, help="Your current age")
    sex = st.radio("⚥ Gender", ['male', 'female'], help="Your gender")
    
    # Physical measurements
    st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin-top: 2rem;">📏 Physical Details</h3>', unsafe_allow_html=True)
    height = st.number_input("📐 Height (cm)", min_value=100, max_value=250, value=170, help="Your height in centimeters")
    weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70, help="Your weight in kilograms")
    
    # Calculate and display BMI
    bmi = round(weight / ((height / 100) ** 2), 1)
    st.markdown(f"""
        <div class="bmi-display">
            <h3>BMI: {bmi}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Lifestyle factors
    st.markdown('<h3 style="color: rgba(255, 255, 255, 0.9); margin-top: 2rem;">👨‍👩‍👧‍👦 Family & Lifestyle</h3>', unsafe_allow_html=True)
    children = st.selectbox("👶 Number of children", ['0', '1', '2', '3', '4', '5', '6', '7+'], help="Number of dependents")
    smoker = st.radio("🚭 Smoking status", ['no', 'yes'], help="Do you smoke?")
    region = st.selectbox("🗺️ Region", ['southeast', 'southwest', 'northeast', 'northwest'], help="Your residential region")
    
    # Create person dataframe
    person_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])
    
    st.markdown('<h3 class="section-header">📋 Your Data Summary</h3>', unsafe_allow_html=True)
    st.dataframe(person_df, hide_index=True, use_container_width=True)
    
    # Add controls in sidebar
    st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Update", key="sidebar_rerun", help="Recalculate with new values", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("📊 Analyze", key="analyze_button", help="Get detailed analysis", use_container_width=True):
            st.success("Analysis updated!")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- MAIN CONTENT --------------------
# Load model and data
model = get_model() if PYCARET_AVAILABLE else None
all_df = get_all_participants()

# Make prediction
if PYCARET_AVAILABLE and model is not None:
    try:
        predicted_charge_id = predict_model(model, data=person_df)["prediction_label"].values[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        predicted_charge_id = predict_insurance_charge(person_df)
else:
    predicted_charge_id = predict_insurance_charge(person_df)

# Display prediction result
st.markdown(f"""
    <div class="prediction-result">
        <h2>Your Estimated Insurance Charge</h2>
        <h1>${predicted_charge_id:,.2f}</h1>
    </div>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h2 class="section-header">📊 Sample Data from Database</h2>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    sample_data = all_df.sample(10)
    st.dataframe(sample_data, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h2 class="section-header">💡 Personalized Recommendations</h2>', unsafe_allow_html=True)
    
    # SHAP Analysis (if available)
    if PYCARET_AVAILABLE and SHAP_AVAILABLE and model is not None:
        try:
            train_features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            person_df_model = person_df[train_features]
            
            # Transform and analyze
            X_transformed = model[:-1].transform(person_df_model)
            gb_model = model.named_steps['actual_estimator']
            explainer = shap.Explainer(lambda x: gb_model.predict(x), X_transformed)
            shap_values = explainer(X_transformed)
        except Exception as e:
            st.warning(f"SHAP analysis unavailable: {e}")
    
    # Convert data types
    person_df_copy = person_df.copy()
    person_df_copy["children"] = person_df_copy["children"].astype(int)
    person_df_copy["bmi"] = person_df_copy["bmi"].astype(float)
    person_df_copy["age"] = person_df_copy["age"].astype(int)
    
    # Generate suggestions
    suggestions = []
    if person_df_copy["smoker"].iloc[0] == "yes":
        suggestions.append("🚭 Quitting smoking could significantly reduce your premium")
    if person_df_copy["bmi"].iloc[0] > 25:
        suggestions.append("🏃‍♂️ Lowering BMI through diet or exercise may reduce your premium")
    if person_df_copy["bmi"].iloc[0] < 18.5:
        suggestions.append("⚖️ Consider maintaining a healthier weight range")
    suggestions.append("📅 Age is fixed, but younger individuals usually pay less")
    if person_df_copy["children"].iloc[0] > 0:
        suggestions.append("👨‍👩‍👧‍👦 More children can sometimes increase premiums - plan accordingly")
    suggestions.append(f"🗺️ Different regions have varying risk factors (current: {person_df_copy['region'].iloc[0]})")
    
    # Display suggestions
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for suggestion in suggestions:
        st.markdown(f"""
            <div class="suggestion-item">
                {suggestion}
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- ADDITIONAL INSIGHTS --------------------
st.markdown('<h2 class="section-header">📈 Market Analysis</h2>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    avg_charge = all_df['charges'].mean()
    st.metric("💰 Average Market Charge", f"${avg_charge:,.2f}")
    comparison = "above" if predicted_charge_id > avg_charge else "below"
    difference = abs(predicted_charge_id - avg_charge)
    st.write(f"Your estimate is **${difference:,.2f} {comparison}** the market average")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    same_age_group = all_df[(all_df['age'] >= age-5) & (all_df['age'] <= age+5)]
    age_group_avg = same_age_group['charges'].mean()
    st.metric("🎂 Your Age Group Average", f"${age_group_avg:,.2f}")
    st.write(f"Ages {age-5}-{age+5}")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    same_region = all_df[all_df['region'] == region]
    region_avg = same_region['charges'].mean()
    st.metric("🗺️ Your Region Average", f"${region_avg:,.2f}")
    st.write(f"Region: {region.title()}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
    <div class="glass-card" style="text-align: center; margin-top: 3rem;">
        <p style="color: rgba(30, 41, 59, 0.6); margin: 0;">
            💡 <strong>Disclaimer:</strong> This is an AI-powered estimate for informational purposes only. 
            Actual insurance charges may vary based on additional factors and company policies.
        </p>
    </div>
""", unsafe_allow_html=True)