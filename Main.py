import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

st.title("ML Model Battle")

test_size = st.slider("Test size (ratio of test set):", 0.1, 0.5, 0.2, 0.05)
random_state = st.number_input("Random state (seed):", min_value=0, max_value=1000, value=42, step=1)
player_model = st.selectbox("Choose your model:", ["LogisticRegression", "RandomForest", "KNN", "SVM", "GradientBoosting"])

if st.button("Fight!"):
    data = pd.read_csv("data.csv")
    data.workclass.replace({'?': 'Others'}, inplace=True)
    data.occupation.replace({'?': 'Others'}, inplace=True)
    data = data[data['workclass'] != 'Without-pay']
    data = data[data['workclass'] != 'Never-worked']
    data = data[(data['age'] <= 75) & (data['age'] >= 17)]
    data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]
    data = data.drop(columns=['education'])

    encoder = LabelEncoder()
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']:
        data[col] = encoder.fit_transform(data[col])

    x = data.drop(columns=['income'])
    y = data['income']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    cpu_model = random.choice([m for m in models if m != player_model])
    st.subheader(f"Player: {player_model}  |  CPU: {cpu_model}")

    results = {}

    for name in [player_model, cpu_model]:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', models[name])
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        if name == player_model:
            st.subheader("Player Model Report:")
            st.markdown(f"**Accuracy:** `{acc:.4f}`")
            st.code(classification_report(y_test, y_pred))
        else:
            st.subheader("CPU Model Report:")
            st.markdown(f"**Accuracy:** `{acc:.4f}`")
            st.code(classification_report(y_test, y_pred))

    if results[player_model] > results[cpu_model]:
        st.success("You Win!")
    elif results[player_model] < results[cpu_model]:
        st.error("CPU Wins!")
    else:
        st.info("It's a Draw!")

