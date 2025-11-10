# app.py
# Streamlit 垃圾郵件分類 Demo

import streamlit as st
import joblib

# 載入模型與向量器
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 垃圾郵件分類器 Demo")
st.write("輸入一段郵件內容，系統會判斷是否為垃圾郵件。")

user_input = st.text_area("請輸入郵件內容：")

if st.button("開始判斷"):
    if user_input.strip() == "":
        st.warning("請輸入文字！")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("這封郵件可能是垃圾郵件 🚨")
        else:
            st.success("這封郵件應該是正常郵件 ✅")
