---
title: Sentiment Chat App
sdk: streamlit
colorFrom: green
colorTo: purple
description: Sentiment-aware chatbot with word highlights.
tags:
  - llm
  - chatbot
  - sentiment-analysis
  - streamlit
  - python
  - cpu-inference
  - google-gemini
  - bert
---

# 💬 Sentiment Chat App

---

* **Demo Hugging Face Space:** [**Sentiment Chat App**](https://huggingface.co/spaces/lupinerh/sentiment-chat-app)

---

This application is an interactive chatbot that:
*   Analyzes the sentiment of user messages in real-time using a user-selectable sentiment analysis model.
*   Visualizes word importance for sentiment determination.
*   Engages in dialogue using a LLM.
*   Offers a choice between a locally run LLM and a cloud-based LLM for dialogue generation.
*   Considers the detected sentiment of the user's message when generating a response.


## Data Sources and Models

*   **Dataset for Sentiment Analysis Training:**
    *   The sentiment analysis model was trained on the **RuTweetCorp dataset**.

*   **Sentiment Analysis Models (User Selectable):**
    *  **ML Model** 
    *  **BERT-based Model:** Fine-tuned [**RuBERT-tiny2**] (https://huggingface.co/seara/rubert-tiny2-russian-sentiment).

*   **Supported LLM Models for Dialogue Generation::**
    *   The application can be configured to use one of the following LLMs:
        *   **Local LLM (CPU-based):**
            *   **Model:** **Phi-3-mini-4k-instruct** by Microsoft.
            *   **Details:** This model runs locally on your CPU.
            *   **Model Card:** [microsoft/Phi-3-mini-4k-instruct-gguf on Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
        *   **Cloud-based LLM (API):**
            *   **Service:** **Google Gemini API**
            *   **Model (example):** `gemini-2.0-flash-lite` (configurable, see `config.yaml`)
            *   **Details:** Utilizes Google's powerful Gemini models via an API. Requires an API key.
            *   **More Info:** [Google AI Gemini API](https://ai.google.dev/docs/gemini_api_overview)
