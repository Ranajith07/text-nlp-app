# app.py

import streamlit as st
from transformers import pipeline

# Optional: Set page layout
st.set_page_config(page_title="NLP App", page_icon="🧠", layout="centered")

# App Title
st.title("🧠 Text Analysis & Generation App")
st.markdown("Choose a task from the sidebar to get started with powerful NLP tools powered by Hugging Face 🤗.")

# Task selection in sidebar
task = st.sidebar.selectbox(
    "Choose NLP Task",
    (
        "Sentiment Analysis",
        "Named Entity Recognition (NER)",
        "Text Generation",
        "Summarization",
        "Translation (English to French)",
        "Question Answering",
        "Grammar & Spelling Correction",
        "Visual Question Answering (VQA)"
        
    )
)

# Common input only for relevant tasks
if task not in ["Question Answering", "Grammar & Spelling Correction","Visual Question Answering (VQA)"]:
    user_input = st.text_area("Enter your text here:")

# Step-by-step explanation toggle
show_steps = st.checkbox("Show Step-by-Step Explanation")

# Define pipeline and run
if task == "Sentiment Analysis":
    if st.button("Analyze Sentiment"):
        nlp = pipeline("sentiment-analysis")
        result = nlp(user_input)
        st.json(result)

        if show_steps:
            st.markdown("""
            ### 🧠 How it works:
            1. Text is tokenized into tokens.
            2. Tokens are passed into a transformer-based classification model.
            3. Model returns sentiment label (POSITIVE/NEGATIVE) with confidence score.
            """)

elif task == "Named Entity Recognition (NER)":
    if st.button("Extract Entities"):
        nlp = pipeline("ner", grouped_entities=True)
        result = nlp(user_input)
        st.json(result)

        if show_steps:
            st.markdown("""
            ### 🧠 How it works:
            1. Text is tokenized into words and subwords.
            2. Each token is classified (e.g., PERSON, LOCATION).
            3. Tokens with the same entity are grouped into full names or phrases.
            """)

elif task == "Text Generation":
    max_len = st.slider("Max Output Length", min_value=20, max_value=200, value=50)
    if st.button("Generate Text"):
        nlp = pipeline("text-generation")
        result = nlp(user_input, max_length=max_len, do_sample=True)
        st.write("### ✍️ Generated Text:")
        st.write(result[0]["generated_text"])

        if show_steps:
            st.markdown("""
            ### 🧠 How it works:
            1. Your text prompt is tokenized.
            2. Model generates the next most likely words (autocompletion).
            3. Tokens are decoded back to human-readable text.
            """)

elif task == "Summarization":
    if st.button("Summarize Text"):
        nlp = pipeline("summarization")
        result = nlp(user_input, max_length=100, min_length=30, do_sample=False)
        st.write("### 📄 Summary:")
        st.write(result[0]["summary_text"])

        if show_steps:
            st.markdown("""
            ### 🧠 How it works:
            1. Input text is encoded using a transformer encoder.
            2. A decoder generates a shorter version of the text.
            3. Output is cleaned and presented as a summary.
            """)

elif task == "Translation (English to French)":
    if st.button("Translate"):
        nlp = pipeline("translation_en_to_fr")
        result = nlp(user_input)
        st.write("### 🇫🇷 Translated Text:")
        st.write(result[0]["translation_text"])

        if show_steps:
            st.markdown("""
            ### 🧠 How it works:
            1. English text is tokenized and passed through an encoder.
            2. A French decoder generates equivalent French phrases.
            3. Tokens are decoded into the final translated text.
            """)

elif task == "Question Answering":
    st.write("### ❓ Question Answering")
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context paragraph:")
    if st.button("Get Answer"):
        if not question or not context:
            st.warning("Please enter both a question and a context.")
        else:
            nlp = pipeline("question-answering")
            result = nlp(question=question, context=context)
            st.json(result)

            if show_steps:
                st.markdown("""
                ### 🧠 How it works:
                1. The question and context are combined and tokenized.
                2. The model identifies the span of text in the context that answers the question.
                3. Returns the answer with its position and score.
                """)
elif task == "Grammar & Spelling Correction":
    st.write("### 📝 Grammar & Spelling Corrector")
    text = st.text_area("Enter a sentence or paragraph with grammatical errors:")

    if st.button("Correct Grammar"):
        if text:
            with st.spinner("Correcting grammar..."):
                corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
                result = corrector(text)
                st.success("✅ Corrected Text:")
                st.write(result[0]['generated_text'])

                if show_steps:
                    st.markdown("""
                    ### 🧠 How it works:
                    1. Your input is treated as a "text-to-text" problem.
                    2. It is passed to a T5 model fine-tuned on grammatically incorrect vs. correct pairs.
                    3. The model outputs the corrected version of your input.
                    """)
        else:
            st.warning("Please enter some text.")
            
elif task == "Visual Question Answering (VQA)":
    st.write("### 🖼️ Upload an image and ask a question about it.")

    uploaded_image = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    question = st.text_input("Ask a question about the image:")

    if uploaded_image and question:
        from PIL import Image
        image = Image.open(uploaded_image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Get Answer"):
            with st.spinner("Analyzing image and question..."):
                vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
                result = vqa_pipeline(image=image, question=question)
                st.success("✅ Answer:")
                st.write(result[0]['answer'])

                if show_steps:
                    st.markdown("""
                    ### 🧠 How it works:
                    1. The image is converted into visual embeddings using a Vision Transformer (ViT).
                    2. The question is encoded as text.
                    3. Both are processed together by a VQA model (ViLT) to generate an answer.
                    """)
    else:
        st.info("Please upload an image and type a question.")


# Footer
st.markdown("---")
st.markdown("Built with 🤗 [Hugging Face Transformers](https://huggingface.co/transformers/) and [Streamlit](https://streamlit.io/).")
