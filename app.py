import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

pipe_lr = joblib.load("model/text_emotion.pkl")

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])


def main():
    st.set_page_config(page_title="Emotion Detector", layout="centered")
    st.markdown(
        "<h1 style='text-align: center; color: #4B8BBE;'>🧠 Text Emotion Detector</h1>",
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center;'>Type something and discover the emotion behind your words!</p>", unsafe_allow_html=True)

    
    with st.sidebar:
        st.title("Emotion Detection App 💬")
        st.markdown("Built with Streamlit, scikit-learn, and 🧠 ML.")
        st.markdown("---")
        st.markdown("📌 Example: `I'm feeling happy today!`")

  
    with st.form(key='emotionForm'):
        raw_text = st.text_area("Your Message", height=150)
        submit = st.form_submit_button("Analyze")

    if submit and raw_text.strip() != "":
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.subheader("🎯 Prediction")
            st.markdown(f"<div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>"
                        f"<h2 style='color: #1f77b4;'>{emotions_emoji_dict.get(prediction, '')} {prediction.upper()}</h2>"
                        f"<p><strong>Confidence:</strong> {np.max(probability):.2f}</p>"
                        f"</div>", unsafe_allow_html=True)

        with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

        st.markdown("---")
        st.success("✅ Emotion analysis completed!")

if __name__ == "__main__":
    main()
