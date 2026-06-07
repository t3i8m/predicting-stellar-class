import streamlit as st
import joblib
import pandas as pd

pipeline = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

st.title("🌌 Stellar Object Classifier")
st.write("Enter object parameters to classify as Galaxy, Star or Quasar")

col1, col2 = st.columns(2)

with col1:
    redshift = st.slider("Redshift", 0.0, 7.0, 0.5)
    u = st.number_input("U filter", value=22.0)
    g = st.number_input("G filter", value=21.0)
    
with col2:
    r = st.number_input("R filter", value=20.0)
    i = st.number_input("I filter", value=19.0)
    z = st.number_input("Z filter", value=19.0)

spectral_type = st.selectbox("Spectral Type", ['M', 'G/K', 'A/F', 'O/B'])
galaxy_population = st.selectbox("Galaxy Population", ['Blue_Cloud', 'Green_Valley', 'Red_Sequence'])

if st.button("🔭 Classify!"):
    input_df = pd.DataFrame([{'alpha': 180.0, 'delta': 30.0,'u': u, 'g': g, 'r': r, 'i': i, 'z': z,'redshift': redshift,'spectral_type': spectral_type,'galaxy_population': galaxy_population}])
    
    pred = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]
    label = le.inverse_transform([pred])[0]
    
    emoji = {'GALAXY': '🌌', 'STAR': '⭐', 'QSO': '✨'}
    st.success(f"Predicted: **{emoji[label]} {label}**")
    
    proba_df = pd.DataFrame({'Class': le.classes_,'Probability': proba})
    st.bar_chart(proba_df.set_index('Class'))