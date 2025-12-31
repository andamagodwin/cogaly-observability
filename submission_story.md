## 💡 Inspiration
Alzheimer’s disease is one of the most complex challenges in modern medicine, characterized by slow onset and subtle early symptoms. While researching machine learning applications in healthcare, I realized something critical: **building a high-accuracy model is only half the battle.**

In clinical settings, "black box" AI models are dangerous. If a model predicts a high risk of Alzheimer's but can't explain why, a doctor cannot trust it. Furthermore, models "drift" over time—a model trained today might fail silently tomorrow due to changing patient demographics or data quality issues.

This inspired me to build **Cogaly**: not just a prediction engine, but a **safety-first observability system**. My goal was to answer the question: *"If this AI begins to fail or behave unsafely, will we know before it harms a patient?"*

## ⚙️ How We Built It
We approached Cogaly as a production-grade engineered system, not just a data science notebook.

### 1. The Core Engine (XGBoost + SHAP)
We used **XGBoost** for the core classification task because of its superior performance on tabular clinical data. To mitigate the "black box" problem, we integrated **SHAP (SHapley Additive exPlanations)** directly into the inference pipeline.

Each prediction $f(x)$ is decomposed into the sum of feature contributions $\phi_i$, giving clinicians an exact reasoning for every risk score:

$$ f(x) = \phi_0 + \sum_{i=1}^M \phi_i $$

Where $\phi_i$ represents the marginal contribution of feature $i$ (e.g., MMSE score or Hippocampal volume) to the final risk probability.

### 2. The Observability Layer (Datadog)
This is the heart of Cogaly. Instead of just logging "200 OK", we emit custom metrics for every prediction:
*   **Confidence Scores**: Tracking the model's certainty ($P(y|x)$).
*   **Data Drift**: Monitoring distributions of key inputs like Age and MMSE.
*   **Latency**: Ensuring real-time responsiveness.

We configured Datadog monitors to alert us on specific "Medical Safety" violations—for example, if the model confidently predicts "Low Risk" for a patient with severe cognitive decline indicators.

### 3. Application & Deployment
*   **Backend**: Python **FastAPI** for high-performance inference.
*   **Frontend**: A responsive, accessible web UI to democratize access.
*   **Deployment**: Hosted on **Render**, auto-deploying from GitHub to ensure continuous delivery.

## 🚧 Challenges We Faced
The journey wasn't smooth. Our biggest hurdle was the **"Works on My Machine"** adaptation of the deployment:

1.  **Serialization Hell**: We faced cryptic `pickle` serialization errors when deploying to Azure due to a Python version mismatch (3.11 vs 3.10). The model trained locally refused to load in the cloud. We solved this by standardizing our runtime environment and migrating to **Render** for more granular control over the build process.
2.  **Explainability vs. Noise**: Initially, SHAP returned contributions for all 30+ features, overwhelming the user. We had to implement a filtering logic to present only the *top 5 statistically significant* factors, translating raw math into clinical insights.

## 🧠 What We Learned
*   **Observability is not optional**: In healthcare AI, observability is safety. Measuring model confidence is just as important as measuring accuracy.
*   **Infrastructure matters**: A great model trapped in a broken container is useless. Mastering the deployment pipeline (Docker, CI/CD, Environment Variables) was a huge learning curve but essential for delivering a real product.
*   **Empathy in UX**: Designing the result screen required empathy—telling someone they are at "High Risk" requires careful framing, clear data, and actionable next steps, not just a raw red banner.

## 🚀 What's Next for Cogaly
We plan to expand Cogaly by introducing **Multimodal Data Support**, incorporating MRI image tensors alongside our tabular data to increase predictive power using ensemble networks.

$$ \hat{y}_{ensemble} = w_1 \cdot P_{tabular} + w_2 \cdot P_{image} $$

We believe Cogaly proves that student hackers can build tools that are not only technically impressive but ethically designed and operationally safe.
