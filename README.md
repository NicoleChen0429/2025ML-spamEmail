# 2025ML-spamEmail

這是一個基於 **Naive Bayes** 模型的簡易垃圾郵件分類專案。  
本專案使用 Python 製作，並透過 Streamlit 提供互動式網頁介面。

---

## 🚀 專案說明

本專案延伸自 Packt Publishing 的《Hands-On Artificial Intelligence for Cybersecurity》  
Chapter 3 - Spam Email 問題範例，並擴充了：

- 更完整的前處理步驟  
- 視覺化與互動式網頁展示  
- CLI 與 Streamlit 雙介面展示方式  

## 📂 專案結構

spam_classifier.py # 模型訓練與測試
app.py # Streamlit 主程式
requirements.txt # 套件清單

## 🧠 使用說明

1. 安裝必要套件：
   ```bash
   pip install -r requirements.txt
2.訓練模型：
   bash
   複製程式碼
   python spam_classifier.py
3.啟動 Streamlit 網頁：
   bash
   複製程式碼
   streamlit run app.py

