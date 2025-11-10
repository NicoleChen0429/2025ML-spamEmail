# spam_classifier.py
# 簡易垃圾郵件分類模型 (Naive Bayes)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 載入資料集（可改為自己的 CSV）
# 範例使用簡單資料
data = {
    "text": [
        "Win a free iPhone now",
        "Limited offer just for you",
        "Meeting schedule update",
        "Let's have lunch tomorrow",
        "You have won a lottery prize"
    ],
    "label": ["spam", "spam", "ham", "ham", "spam"]
}
df = pd.DataFrame(data)

# 2. 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)

# 3. 向量化文字
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. 建立模型
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. 測試模型
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("準確率:", acc)
print("混淆矩陣:\n", cm)

# 6. 儲存模型與向量器
import joblib
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("模型已訓練完成並儲存！")
