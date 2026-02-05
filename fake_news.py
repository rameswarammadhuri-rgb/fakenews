import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

st.title("📰 Fake News Detection System")
st.write("This app predicts whether a news article is Fake or Real using Machine Learning.")

@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv", encoding="utf-8", engine="python", on_bad_lines="skip")
    true = pd.read_csv("True.csv", encoding="utf-8", engine="python", on_bad_lines="skip")

    fake = fake.rename(columns={fake.columns[0]: "text"})
    true = true.rename(columns={true.columns[0]: "text"})

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], axis=0)
    data = data[["text", "label"]]
    data = data.dropna()
    return data

data = load_data()

# Split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize text
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Train classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"{round(score*100, 2)}%")

# User input
user_input = st.text_area("Enter news text to check:")
if st.button("Check News"):
    user_vec = tfidf_vectorizer.transform([user_input])
    prediction = pac.predict(user_vec)[0]
    if prediction == 0:
        st.error("❌ This news is likely **Fake**")
    else:
        st.success("✅ This news is likely **Real**")
