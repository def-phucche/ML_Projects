import streamlit as st
import pandas as pd
import pickle


st.title("This is a news classifier")
text = st.text_area("Enter news:")
df = pd.DataFrame({
    'text':[text]
})


if st.button('Predict'):
    with open('nlp.pickle','rb') as f:
        model = pickle.load(f)
    
    ans = model.predict(df['text'])
    st.success(ans[0])