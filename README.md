# Cogaly Observability

**Cogaly Observability** is a healthcare-focused AI observability and safety monitoring system built for the Datadog × Google Cloud hackathon. The project demonstrates how to design **end-to-end observability for an LLM-powered medical AI application**, with a specific focus on *early Alzheimer’s risk assessment*.

Rather than optimizing for raw medical accuracy, Cogaly focuses on a critical and often overlooked problem: **LLMs and ML models can fail silently**. In healthcare, these silent failures—such as confidence collapse, data drift, or explanation instability—can have serious consequences.

Cogaly combines:
- A **structured Alzheimer’s risk model** trained on synthetic clinical data
- An **LLM reasoning layer powered by Gemini on Vertex AI** for natural language explanations
- **Datadog observability** to monitor runtime health, model quality signals, and safety indicators

This repository contains the full, reproducible implementation of Cogaly, including model training, application code, Datadog configurations, traffic generation, and deployment instructions.

---

## 🚀 Project Overview

Healthcare AI systems are increasingly deployed in real-world environments, yet most monitoring strategies only track infrastructure metrics like latency and error rates. These metrics are insufficient for medical AI systems, where **model confidence degradation, drift, and hallucinations** can occur without triggering traditional alerts.

**Cogaly** addresses this gap by treating an AI model as a *production system that must be observed, audited, and acted upon*.

The system is designed to answer one key question:

> *If this AI system begins to behave unsafely, will engineers and clinicians know—and know what to do next?*

---

## 🧠 System Architecture

```
User Input (Structured + Text)
        ↓
Cogaly Core Model (Tabular ML Risk Scoring)
        ↓
Gemini (Vertex AI) – LLM Reasoning & Explanation
        ↓
API Response
        ↓
Telemetry → Datadog (Metrics, Logs, Events, Incidents)
```

### Key Components
- **Cogaly Core Model**: A machine learning model trained on structured Alzheimer’s-related clinical and lifestyle data. Outputs a risk score and confidence signals.
- **LLM Layer (Gemini)**: Generates natural-language explanations of model outputs while explicitly expressing uncertainty and avoiding diagnostic claims.
- **Observability Layer (Datadog)**: Captures runtime metrics, LLM behavior, quality signals, and safety indicators, and triggers actionable incidents.

---

## 🧪 Dataset

Cogaly is trained on a **synthetic Alzheimer’s Disease dataset** containing demographic, lifestyle, medical, cognitive, and functional assessment features.

- **Source**: Kaggle – Alzheimer’s Disease Dataset (Rabie El Kharoua, 2024)
- **License**: CC BY 4.0
- **Nature**: Fully synthetic (no real patient data)

This dataset is suitable for educational and research purposes and avoids ethical risks associated with real patient information.

> ⚠️ **Disclaimer**: This project is a research and observability prototype. It is **not a diagnostic tool** and must not be used for clinical decision-making.

---

## 🧠 Model Design (Cogaly)

### Training
- Model training is performed **outside Vertex AI** (e.g., Google Colab or local environment)
- The trained model artifact is exported and versioned (e.g., `cogaly_v1.pkl`)

### Outputs
The Cogaly model outputs structured signals designed for observability:

```json
{
  "risk_score": 0.72,
  "confidence": 0.41,
  "entropy": 1.88,
  "input_anomaly": false
}
```

These signals are intentionally simple yet expressive, enabling downstream detection of unsafe behavior.

---

## 🤖 LLM Integration (Gemini on Vertex AI)

Cogaly uses **Gemini via Vertex AI** as the LLM reasoning layer. The LLM:
- Converts structured risk outputs into clinician-friendly explanations
- Highlights uncertainty and limitations
- Avoids medical diagnosis language

### Why Gemini?
- Fully compliant with hackathon requirements
- Reliable token-level telemetry
- Strong reasoning and summarization capabilities
- Cost-efficient for limited cloud credits

---

## 📊 Observability Strategy (Datadog)

Cogaly treats observability as a first-class feature.

### Telemetry Collected
- **Runtime**: latency, errors, request volume
- **LLM Metrics**: token counts, response length, latency
- **Model Quality Signals**:
  - Confidence decay
  - Entropy trends
  - Drift proxies
  - High-risk/low-confidence combinations

### Detection Rules
At least three Datadog monitors are defined, including:
1. High Alzheimer’s risk predictions with low model confidence
2. Sustained confidence degradation over time
3. Abnormal LLM token usage or explanation instability

Each detection rule automatically creates an **actionable incident** in Datadog with contextual information and suggested remediation steps.

---

## 🚨 Incident Management

When unsafe behavior is detected, Datadog automatically:
- Opens an incident or case
- Attaches relevant telemetry (model version, metrics, sample inputs)
- Links to a runbook with recommended actions, such as:
  - Escalation to human review
  - Model rollback
  - Threshold adjustment
  - Traffic throttling

This ensures that issues are not only detected, but **actionable**.

---

## 🧰 Repository Structure

```text
cogaly-observability/
├── app/              # API, Gemini integration, telemetry emission
├── model/            # Training notebook and model artifacts
├── datadog/          # Dashboards, monitors, SLOs, runbooks (JSON exports)
├── traffic/          # Traffic generator to trigger detection rules
├── deploy/           # Google Cloud & Vertex AI setup guides
├── README.md
├── LICENSE
└── requirements.txt
```

---

## ☁️ Deployment

- **Model Hosting**: Google Cloud Vertex AI
- **LLM**: Gemini (Vertex AI)
- **Application**: Python (FastAPI or Flask)
- **Observability**: Datadog (full access via GitHub Student Pack)

Detailed deployment instructions are provided in the `deploy/` directory.

---

## 📦 Datadog Assets

This repository includes:
- JSON exports of dashboards
- Monitor and SLO configurations
- Incident runbooks

These assets can be imported directly into a Datadog organization for reproducibility.

---

## 🎥 Demo

A 3-minute demo video accompanies this submission and demonstrates:
- The Cogaly architecture
- Live telemetry streaming to Datadog
- Detection rules triggering incidents
- Actionable context for AI engineers

---

## 🧾 License

This project is open-source and released under an OSI-approved license. See the `LICENSE` file for details.

---

## 🧠 Final Note

Cogaly is not about building the most accurate medical model—it is about building **safe, observable, and responsible AI systems**. In domains like healthcare, observability is not optional; it is essential.

This project demonstrates how Datadog and Google Cloud can be combined to make AI systems transparent, accountable, and operationally safe.

