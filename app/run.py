import time
import requests
import subprocess

import pandas as pd
import streamlit as st
import altair as alt
from streamlit.runtime.state import SessionState


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def predict(sentence):
    url = "http://localhost:8080/predictions/text_classifier_endpoint"
    headers = {"Content-Type": "application/json"}
    data = {"data": sentence}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response = response.json()
        idx2label = {0: "Negative", 1: "Positive"}
        predicted_class_label = idx2label[response["predicted_class"]]
    else:
        return [""], 0.0

    return [predicted_class_label], response["pred_per_class"]


predefined_categories = ["Negative", "Positive"]

st.set_page_config(page_title="Sentence Classification", layout="centered", initial_sidebar_state="expanded")
local_css("styles.css")
st.markdown("### Sentence Classification")




predictions = []

sentence = st.text_area(
    label="Enter a sentence",
    value="This movie had the best acting and the dialogue was so good. I loved it.",
    max_chars=200,
    disabled=True,
)

# --------------------------------------- Main --------------------------------------- #

if st.button("Submit"):
    with st.spinner("Predicting..."):
        predictions, probs = predict(sentence)

    prediction_tags = [f'<span class="rounded-tag">{prediction}</span>' for prediction in predictions]
    predefined = [category for category in predefined_categories if category not in predictions]
    predefined_tags = [f'<span class="rounded-tag-disabled">{category}</span>' for category in predefined]

    st.divider()
    st.markdown("Predicted label")
    prediction_tags = " ".join(prediction_tags + predefined_tags)
    st.write(prediction_tags, unsafe_allow_html=True)

    st.markdown("Prediction Probabilities")
    prob_df = pd.DataFrame({"Prediction": predefined_categories, "Probability": map(lambda x: round(x*100, 2), probs)})
    prob_df = prob_df.sort_values(by="Probability", ascending=False)

    chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X("Prediction:N", sort=None),                          # Nominal data
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 100])),  # Quantitative
        tooltip=["Prediction", "Probability"],
    )
    st.altair_chart(chart, use_container_width=True)


# --------------------------------------- Sidebar --------------------------------------- #

with st.sidebar:
    start_server = st.button("Start Server", use_container_width=True)
    stop_server = st.button("Stop Server", use_container_width=True)

    if start_server:
        with st.spinner("Starting server..."):
            subprocess.Popen([
                "torchserve", "--start", "--ncs", "--model-store", "../model_store", "--models",
                "text_classifier_endpoint=text_classifier_endpoint.mar"
            ])
            time.sleep(5)
            st.success("Server started successfully!")

    if stop_server:
        with st.spinner("Stopping server..."): subprocess.Popen(["torchserve", "--stop"])
        time.sleep(2)
        st.success("Server stopped successfully!")