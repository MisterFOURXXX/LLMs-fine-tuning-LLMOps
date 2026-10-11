# Fine-Tuning LLMs - Production ML Engineering Pipeline (LLMOps)

> A production-ready LLMOps pipeline for fine-tuning Qwen3-0.6B with Direct Preference Optimization (DPO) and QLoRA to improve code-generation quality. It covers the full lifecycle: DPO preference-pair construction, 4-bit fine-tuning with LoRA, MLflow experiment tracking on DagsHub, DVC data/model versioning, multi-metric evaluation (BLEU, ROUGE, BERTScore, perplexity, code-quality proxies), Dockerized FastAPI inference, and AWS deployment with ECR, Lambda/API Gateway, and Terraform.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.57-yellow.svg)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/peft-%3E%3D0.6-8a2be2.svg)](https://huggingface.co/docs/peft)
[![TRL](https://img.shields.io/badge/trl-%3E%3D0.7-ff6f61.svg)](https://huggingface.co/docs/trl)
[![bitsandbytes](https://img.shields.io/badge/bitsandbytes-%3E%3D0.48-8a2be2.svg)](https://github.com/TimDettmers/bitsandbytes)
[![MLflow](https://img.shields.io/badge/mlflow-%3E%3D2.8-0194e2.svg)](https://mlflow.org/)
[![DagsHub](https://img.shields.io/badge/dagshub-%3E%3D0.3-ff6f00.svg)](https://dagshub.com/)
[![DVC](https://img.shields.io/badge/dvc-%3E%3D3.0-13adc7.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/docker-%3E%3D24.0-2496ed.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/aws-ecr%20%7C%20lambda%20%7C%20terraform-ff9900.svg)](https://aws.amazon.com/)
[![FastAPI](https://img.shields.io/badge/fastapi-%3E%3D0.100-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Project Overview

A production-ready LLMOps pipeline for fine-tuning Qwen3-0.6B using Direct Preference Optimization (DPO) with QLoRA, and deploying a containerized inference service to AWS. This pipeline includes experiment tracking with MLflow on DagsHub, data version control with DVC, containerization with Docker, and comprehensive model evaluation for code-generation quality.

This project is a production-oriented LLMOps pipeline for fine-tuning Qwen3-0.6B using DPO + QLoRA. It is designed for ML engineers who need to take a small language model from raw data through training, evaluation, packaging, and deployment. The pipeline covers the full lifecycle of a production language model service:

- Raw data ingestion and preprocessing.
- DPO preference-pair construction.
- Efficient 4-bit fine-tuning with LoRA adapters.
- Experiment tracking with MLflow on DagsHub.
- Data and model versioning with DVC.
- Multi-metric evaluation with code-quality proxies.
- Dockerized FastAPI inference.
- AWS deployment with ECR, Lambda/API Gateway, and Terraform.
- Automated tests and monitoring.

Fine-tuning large language models is expensive and operationally complex. This project demonstrates how parameter-efficient fine-tuning and MLOps best practices can make small-model adaptation reproducible, observable, and deployable at production quality. It shows how an ML engineer can build a defensible pipeline for aligning a code-oriented model while keeping costs, latency, and reliability under control.

### Engineering Goals

- **Reproducibility:** every hyperparameter lives in version-controlled YAML, every artefact is DVC-tracked, and every run is MLflow-tracked.
- **Modularity:** data, training, evaluation, tracking, and deployment are independent modules with documented interfaces.
- **Deployability:** the same trained model is served via a Dockerized FastAPI service and deployable to AWS via Terraform.
- **Observability:** MLflow metrics, DVC pipeline status, CloudWatch logs, and API health checks give full visibility into training and serving.
- **Cost control:** QLoRA + 4-bit quantisation allow fine-tuning on a single consumer or workstation GPU (≥ 8 GB VRAM).
- **Testability:** data processing, model loading, and API behaviour each have pytest coverage with a coverage report.

### Table of Contents

- [Project Overview](#project-overview)
- [1. Introduction](#1-introduction)
- [2. Foundation](#2-foundation)
  - [2.1 Direct Preference Optimization](#21-direct-preference-optimization)
  - [2.2 QLoRA and Parameter-Efficient Fine-Tuning](#22-qlora-and-parameter-efficient-fine-tuning)
  - [2.3 Preference-Pair Construction from Scored Data](#23-preference-pair-construction-from-scored-data)
  - [2.4 Experiment Tracking and Data Versioning](#24-experiment-tracking-and-data-versioning)
  - [2.5 Reproducibility Engineering](#25-reproducibility-engineering)
  - [2.6 Intended Scope and Operational Limits](#26-intended-scope-and-operational-limits)
- [3. System Design](#3-system-design)
  - [3.1 Design](#31-design)
  - [3.2 Core Components](#32-core-components)
  - [3.3 High-Level Workflow](#33-high-level-workflow)
  - [3.4 Data Collection and Preprocessing](#34-data-collection-and-preprocessing)
  - [3.5 Model and Training](#35-model-and-training)
  - [3.6 Experiment Tracking and Versioning](#36-experiment-tracking-and-versioning)
  - [3.7 Deployment as a Production Artifact](#37-deployment-as-a-production-artifact)
  - [3.8 Reproducibility Plan](#38-reproducibility-plan)
  - [3.9 Hardware Requirements](#39-hardware-requirements)
- [4. Evaluation](#4-evaluation)
  - [4.1 Evaluation Strategy](#41-evaluation-strategy)
  - [4.2 Experimental Conditions and Ablations](#42-experimental-conditions-and-ablations)
  - [4.3 Metrics](#43-metrics)
- [5. Features](#5-features)
- [6. Project Structure](#6-project-structure)
- [7. Get Started - Step-By-Step](#7-get-started---step-by-step)
  - [7.1 Initial Setup Checklist](#71-initial-setup-checklist)
  - [7.2 Model Experiment](#72-model-experiment)
- [8. Docker Deployment](#8-docker-deployment)
- [9. AWS Deployment](#9-aws-deployment)
- [10. API Usage](#10-api-usage)
- [11. Testing](#11-testing)
- [12. Environment Variables](#12-environment-variables)
- [13. Monitoring and Logging](#13-monitoring-and-logging)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Engineering Considerations and Limitations](#15-engineering-considerations-and-limitations)
- [16. Conclusion](#16-conclusion)
- [17. Contributing](#17-contributing)
- [18. References](#18-references)
- [19. License](#19-license)
- [About](#about)

---

## 1. Introduction

A production-ready LLMOps pipeline for fine-tuning Qwen3-0.6B using Direct Preference Optimization (DPO) with QLoRA, and deploying a containerized inference service to AWS. This pipeline includes experiment tracking with MLflow on DagsHub, data version control with DVC, containerization with Docker, and comprehensive model evaluation for code-generation quality.

This project is a production-oriented LLMOps pipeline for fine-tuning Qwen3-0.6B using DPO + QLoRA. It is designed for ML engineers who need to take a small language model from raw data through training, evaluation, packaging, and deployment. The pipeline covers the full lifecycle of a production language model service:

- Raw data ingestion and preprocessing.
- DPO preference-pair construction.
- Efficient 4-bit fine-tuning with LoRA adapters.
- Experiment tracking with MLflow on DagsHub.
- Data and model versioning with DVC.
- Multi-metric evaluation with code-quality proxies.
- Dockerized FastAPI inference.
- AWS deployment with ECR, Lambda/API Gateway, and Terraform.
- Automated tests and monitoring.

Fine-tuning large language models is expensive and operationally complex. This project demonstrates how parameter-efficient fine-tuning and MLOps best practices can make small-model adaptation reproducible, observable, and deployable at production quality. It shows how an ML engineer can build a defensible pipeline for aligning a code-oriented model while keeping costs, latency, and reliability under control.

### 1.1 Engineering Hypotheses

The pipeline is designed and evaluated against four engineering assumptions that guide the design of the training and deployment paths:

- **H1:** DPO + QLoRA improves preference accuracy and reward margins over the base model.
- **H2:** 4-bit quantization reduces GPU memory use with minimal degradation in BLEU, ROUGE, and BERTScore.
- **H3:** Code-quality proxies (functions, imports, comments, line count) correlate with perceived usefulness better than BLEU alone.
- **H4:** DVC + MLflow enable reliable reproduction of training and evaluation across machines.

These hypotheses frame the ablation surface documented in [§4.2](#42-experimental-conditions-and-ablations) and the metrics tracked throughout the pipeline.

---

## 2. Foundation

This chapter documents the technical foundations that the pipeline is built on, and the rationale for each technology choice.

### 2.1 Direct Preference Optimization

DPO (Rafailov et al., 2023) reframes RLHF as a supervised classification problem on preference pairs, eliminating the need for a separate reward model and a PPO loop. Instead of learning a scalar reward function and then optimising a policy against it, DPO directly optimises the policy to increase the log-probability ratio of the "chosen" response over the "rejected" response, relative to a frozen reference model. This makes alignment stable, cheap, and competitive with RLHF on many tasks, which is why it is the standard alignment method for small, open models.

This pipeline uses DPO with `beta = 0.1` as the final training stage after QLoRA, and constructs preference pairs from Stack Overflow answer scores (see [§2.3](#23-preference-pair-construction-from-scored-data)).

### 2.2 QLoRA and Parameter-Efficient Fine-Tuning

Full fine-tuning of even modest LLMs is prohibitively expensive. PEFT methods such as **LoRA** (Hu et al., 2022) and **QLoRA** (Dettmers et al., 2023) reduce trainable parameters by orders of magnitude while preserving most of the downstream quality.

- **LoRA** injects small trainable matrices into selected linear modules; the base model weights are frozen.
- **QLoRA** combines LoRA with 4-bit NF4 quantisation of the base model, which cuts the memory footprint of the base weights by ~75% and allows fine-tuning on a single GPU.

This pipeline uses QLoRA with `r = 16`, `alpha = 32`, `dropout = 0.1`, and 4-bit NF4 quantisation, consistent with the recommendations of Dettmers et al. (2023). QLoRA is what makes a 0.6B model trainable on a workstation-class GPU (≥ 8 GB VRAM).

### 2.3 Preference-Pair Construction from Scored Data

DPO requires paired preference data of the form `(prompt, chosen, rejected)`. This pipeline constructs those pairs directly from Stack Overflow's answer-score field:

- **Chosen:** the highest-scored answer to a question.
- **Rejected:** the lowest-scored answer to the same question.

The score threshold (`score > 5`) filters out low-quality answers and noise, so that the "chosen" answer is not merely the lesser of two bad options. This is a cheap and effective way to obtain high-signal preference pairs without additional human annotation.

### 2.4 Experiment Tracking and Data Versioning

MLOps practice separates **code**, **data**, and **experiment state**. This pipeline uses:

- **MLflow on DagsHub** for experiment tracking: parameters, metrics, and artifacts logged per run.
- **DVC** for data and model versioning: raw Stack Overflow CSVs, processed datasets, and fine-tuned model artifacts are all DVC-tracked.
- **DVC pipelines** (`dvc repro preprocess`, `dvc repro train`, `dvc repro evaluate`) for staged reproducibility.

This combination mirrors the recommendations of Sculley et al. (2015) on hidden technical debt in ML systems and of the MLOps literature on reproducibility (Garg et al., 2021; Kreuzberger et al., 2023).

### 2.5 Reproducibility Engineering

Reproducibility in LLM fine-tuning is difficult because training is stochastic, quantisation affects numeric behaviour, and data preprocessing choices matter. This pipeline addresses reproducibility through:

- all hyperparameters stored in version-controlled YAML files,
- DVC-tracked raw data, processed datasets, and model artifacts,
- MLflow-tracked parameters, metrics, and artifacts per run,
- Docker images for consistent inference,
- pytest coverage for data processing, model loading, and API behaviour,
- `.env.example` for explicit environment-variable documentation.

The goal is not to claim perfect determinism, but to make the pipeline **auditable and reusable** across machines and deployments.

### 2.6 Intended Scope and Operational Limits

This pipeline is intended for production and educational use: to demonstrate how a small, preference-aligned code model can be trained, evaluated, packaged, and deployed with MLOps discipline. It is **not** intended as a production code assistant without further validation, safety review, and human-in-the-loop processes. The Stack Overflow data is used under its original license; generated code may contain insecure, biased, or outdated patterns. Automatic metrics (BLEU, ROUGE, BERTScore) are proxies for quality and should not replace human evaluation in high-stakes settings.

---

## 3. System Design

### 3.1 Design

This project follows a production-grade, reproducible ML engineering design. The central goal is to fine-tune Qwen3-0.6B with DPO + QLoRA, evaluate it against the base model, and deploy it as a low-latency inference service. The pipeline is designed so that every stage—from raw data to served endpoint—is versioned, tracked, and reproducible.

### 3.2 Core Components

| Component | Purpose |
|---|---|
| Data Pipeline | Cleans Stack Overflow data and builds DPO preference pairs. |
| DPO + QLoRA Training | Fine-tunes Qwen3-0.6B with 4-bit quantization and LoRA adapters. |
| MLflow + DagsHub | Tracks parameters, metrics, and artifacts for every run. |
| DVC | Versions datasets, processed artifacts, and trained models. |
| Evaluation Suite | Measures BLEU, ROUGE, BERTScore, perplexity, and code-quality proxies. |
| Docker Inference API | Serves the fine-tuned model via FastAPI with `/health` and `/generate`. |
| AWS Deployment | ECR, Lambda/API Gateway, and Terraform for infrastructure-as-code. |
| Tests | Pytest coverage for data, model, and API behavior. |
| Monitoring | MLflow metrics, DVC pipeline status, and API health checks. |

### 3.3 High-Level Workflow

```
Stack Overflow CSVs
        |
        V
DVC-tracked raw data
        |
        V
Preprocessing → DPO preference pairs → train/val/test splits
        |
        V
Qwen3-0.6B + 4-bit quantization + LoRA
        |
        V
DPO training
        |
        V
MLflow tracking + DVC artifacts
        |
        V
Evaluation (BLEU, ROUGE, BERTScore, perplexity, code metrics)
        |
        V
Dockerized FastAPI inference
        |
        V
AWS ECR + Lambda/API Gateway (Terraform)
```

### 3.4 Data Collection and Preprocessing

- **Source:** Stack Overflow `stacksample` dataset from Kaggle.
- **Files:** `Questions.csv`, `Answers.csv`, `Tags.csv`.
- **Cleaning:** Remove HTML tags from question and answer bodies.
- **Filtering:** Keep questions with score > 5.
- **Preference Pairs:**
  - **Chosen:** Highest-scored answer.
  - **Rejected:** Lowest-scored answer.
- **Splits:** 80% train, 10% validation, 10% test.
- **Format:** Convert to conversational format suitable for DPO training.
- **Versioning:** Track raw and processed data with DVC.

### 3.5 Model and Training

- **Base Model:** Qwen3-0.6B.
- **Quantization:** 4-bit QLoRA.
- **Adapters:** LoRA applied to selected modules.
- **Objective:** DPO loss for preference alignment.
- **Key Hyperparameters:**
  - `num_train_epochs`: 30
  - `learning_rate`: 5e-5
  - `per_device_train_batch_size`: 2
  - `gradient_accumulation_steps`: 8
  - `beta`: 0.1 (DPO temperature)
  - `max_length`: 1024
- **Hardware:** NVIDIA GPU with at least 8GB VRAM (RTX A4000 recommended).
- **Monitoring:** GPU usage via `nvidia-smi`; training metrics via MLflow.

### 3.6 Experiment Tracking and Versioning

- **MLflow:** Logs parameters, metrics, and artifacts for every training and evaluation run.
- **DagsHub:** Remote MLflow tracking server.
- **DVC:** Versions raw data, processed datasets, and model artifacts.
- **Configuration:** All hyperparameters stored in `config/config.yaml` and `config/params.yaml`.
- **Manual Logging:** Hyperparameters logged without Optuna optimization to keep the pipeline lightweight.

### 3.7 Deployment as a Production Artifact

- **Inference API:** FastAPI service with `/health` and `/generate` endpoints.
- **Containerization:** Docker and Docker Compose for local reproducibility.
- **Cloud Deployment:** AWS ECR for images, Lambda/API Gateway for serverless inference, or ECS for GPU-backed serving.
- **Infrastructure as Code:** Terraform for repeatable AWS provisioning.
- **Purpose:** Production-ready code-generation service.

### 3.8 Reproducibility Plan

- All configurations stored in version-controlled YAML files.
- Raw and processed data tracked with DVC.
- Every run logged to MLflow with parameters, metrics, and artifacts.
- Docker image provided for consistent inference.
- Tests included for data processing, model loading, and API behavior.
- Environment variables documented in `.env.example`.

### 3.9 Hardware Requirements

- **Hardware:** NVIDIA GPU with at least 8GB VRAM (RTX A4000 recommended).
- **CUDA:** 12.4 or higher.
- **Python:** 3.11 or higher.
- **Docker:** 24.0 or higher (for container deployment).
- **AWS CLI:** Configured with appropriate credentials (for AWS deployment).
- **DagsHub Account:** For experiment tracking and DVC remote storage.

---

## 4. Evaluation

### 4.1 Evaluation Strategy

Evaluation is performed on the held-out test set using four complementary metric families:

- **Text similarity:** BLEU, ROUGE-1/2/L, BERTScore F1.
- **Language modelling:** Perplexity.
- **Code-quality proxies:**
  - Presence of functions, imports, and comments.
  - Average function count and line count.
  - Programming language detection (Python, JavaScript, Java, C++, Go, Rust, SQL).
- **Preference metrics:** `rewards/accuracies` and `rewards/margins` from training.

Metrics are saved to `metrics/evaluation_metrics.json` and logged to MLflow.

### 4.2 Experimental Conditions and Ablations

- **Baseline:** Zero-shot Qwen3-0.6B.
- **Main Condition:** DPO + QLoRA.
- **Optional Baseline:** Supervised fine-tuning (SFT).
- **Ablations:**
  - LoRA rank: 4, 8, 16, 32.
  - DPO beta: 0.01, 0.1, 0.5.
  - Max sequence length: 512, 1024.
- **Replication:** Run multiple seeds where compute permits and report mean ± standard deviation.

### 4.3 Metrics

| Category | Metric | Definition |
|---|---|---|
| Text similarity | BLEU | Precision-oriented n-gram overlap with the reference answer |
| Text similarity | ROUGE-1/2/L | N-gram and longest-common-subsequence overlap with the reference |
| Text similarity | BERTScore F1 | Semantic similarity to the reference, harmonic mean of P and R |
| Language modelling | Perplexity | Model confidence on the generated text |
| Code quality | `has_functions_ratio` | Fraction of responses containing at least one function definition |
| Code quality | `has_imports_ratio` | Fraction of responses containing at least one import statement |
| Code quality | `has_comments_ratio` | Fraction of responses containing at least one comment |
| Code quality | `avg_function_count` | Average number of function definitions per response |
| Code quality | `avg_line_count` | Average number of code lines per response |
| Code quality | Language distribution | Detected programming languages across generated responses |
| Preference | `rewards/accuracies` | Share of batch samples where the chosen response scores higher than the rejected |
| Preference | `rewards/margins` | Mean margin between chosen and rejected rewards |
| Performance | Query time, GPU memory, CPU | Serving latency and resource usage per request |

---

## 5. Features

- **DPO Fine-tuning with QLoRA:** Efficient 4-bit quantization + LoRA adapters.
- **Experiment Tracking:** MLflow with DagsHub integration.
- **Data Version Control:** DVC for dataset and model versioning.
- **Hyperparameter Logging:** Manual parameter logging (no Optuna optimization).
- **Comprehensive Metrics:** BLEU, ROUGE, BERTScore, perplexity, code quality metrics.
- **Multi-language Support:** Python, JavaScript, Java, C++, Go, Rust, SQL detection.
- **Docker Containerization:** Easy deployment with Docker Compose.
- **AWS Deployment:** ECR, Lambda, API Gateway with Terraform.
- **Integration Tests:** Pytest with coverage reporting.

---

## 6. Project Structure

```
llmops-rag-pipeline/
├── .gitignore                   # Git ignore rules
├── dvc.yaml                     # DVC pipeline stages
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build instructions
├── docker-compose.yml           # Docker Compose configuration
├── setup_env.sh                 # CUDA and environment setup script
├── config/
│   ├── config.yaml              # Main configuration file
│   └── params.yaml              # Hyperparameter search space (for logging)
├── data/
│   └── raw/                     # Raw CSV files (Questions.csv, Answers.csv, Tags.csv)
├── models/                      # Trained model artifacts
├── notebooks/                   # Jupyter notebooks for development
├── src/                         # Source code
│   ├── __init__.py
│   ├── data/                    # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── dataset_builder.py
│   ├── models/                  # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── utils/                   # Utility functions and metrics
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── helpers.py
│   └── tracking/                # MLflow tracking utilities
│       ├── __init__.py
│       ├── mlflow_tracker.py
│       └── optuna_optimizer.py
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── deployment/                  # Deployment scripts and Terraform configs
│   ├── aws_deploy.py
│   ├── inference_api.py
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
└── scripts/                     # Pipeline execution scripts
    ├── run_pipeline.py          # Main training pipeline
    ├── run_optimization.py      # Hyperparameter logging (manual)
    └── deploy.sh                # Deployment helper script
```

---

## 7. Get Started - Step-By-Step

### 7.1 Initial Setup Checklist

```bash
# 1. Make scripts executable
chmod +x setup_env.sh
chmod +x scripts/*.py
chmod +x deployment/*.py

# 2. Run environment setup
./setup_env.sh

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file from template
cp .env.example .env
# Edit .env with your credentials

# 6. Initialize DVC
dvc init

# 7. Configure DVC remote
source .env
dvc remote add origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user ${DAGSHUB_USER}
dvc remote modify origin --local password ${DAGSHUB_TOKEN}

# 8. Run preprocessing
python scripts/run_preprocessing.py

# 9. Run hyperparameter logging
python scripts/run_optimization.py

# 10. Run training pipeline
python scripts/run_pipeline.py
```

### 7.2 Model Experiment

#### Step 1: Environment Setup

First, run the CUDA environment setup script:

```bash
# Make script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

The `setup_env.sh` script performs the following:

- Removes conflicting NVIDIA packages
- Updates system dependencies
- Cleans apt cache
- Removes conflicting deepspeed packages

#### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

#### Step 3: Configure DagsHub and MLflow

**3.1 Create a DagsHub Repository**

1. Go to dagshub.com and sign up/log in.
2. Create a new repository (e.g., `llmops-rag-pipeline`).
3. Note your username and repository name.

**3.2 Get DagsHub Token**

1. Go to Settings → Tokens.
2. Generate a new token with write permissions.
3. Save the token securely.

**3.3 Set Environment Variables**

```bash
# Set DagsHub authentication
export DAGSHUB_USER="your-dagshub-username"
export DAGSHUB_TOKEN="your-dagshub-token"

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.mlflow"

# Optional: Save to .env file for persistence
echo "DAGSHUB_USER=${DAGSHUB_USER}" >> .env
echo "DAGSHUB_TOKEN=${DAGSHUB_TOKEN}" >> .env
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}" >> .env
```

**3.4 Login to Hugging Face**

```bash
# Install huggingface hub CLI
pip install huggingface-hub

# Login (you'll need an HF token from https://huggingface.co/settings/tokens)
hf login
```

**3.5 Configure DVC Remote**

```bash
# Initialize DVC (if not already done)
dvc init

# Add DagsHub DVC remote
dvc remote add origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc

# Set authentication for the remote
dvc remote modify origin --local auth basic
dvc remote modify origin --local user ${DAGSHUB_USER}
dvc remote modify origin --local password ${DAGSHUB_TOKEN}

# Set as default remote
dvc remote default origin

# Commit DVC config
git add .dvc/config .dvcignore .gitignore
git commit -m "Configure DVC with DagsHub remote"
```

#### Step 4: Prepare Data

**4.1 Download Stack Overflow Dataset**

```bash
# Install kaggle CLI
pip install kaggle

# Configure Kaggle API (place kaggle.json in ~/.kaggle/)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download stackoverflow/stacksample

# Extract files
unzip stacksample.zip -d data/raw/
```

**4.2 Verify Data Files**

Ensure the following files exist in `data/raw/`:

- `Questions.csv` (~1.92 GB)
- `Answers.csv` (~1.61 GB)
- `Tags.csv`

**4.3 Track Data with DVC**

```bash
# Add raw data directory to DVC
dvc add data/raw

# Commit DVC tracking files
git add data/raw.dvc .gitignore
git commit -m "Track raw data with DVC"

# Push data to DagsHub remote
dvc push -r origin
```

#### Step 5: Run Preprocessing

```bash
# Run preprocessing stage
dvc repro preprocess

# Or run directly
python scripts/run_preprocessing.py
```

**What preprocessing does:**

1. Cleans HTML tags from question and answer bodies.
2. Filters questions with score > 5.
3. Creates DPO preference pairs (chosen = highest score answer, rejected = lowest score answer).
4. Splits data into train/validation/test (80/10/10).
5. Converts to conversational format for the model.
6. Saves processed datasets to `data/processed/`.

**Expected output:**

```
Building datasets from CSV files...
Loading questions from data/raw/Questions.csv
Loading answers from data/raw/Answers.csv
Sampling top 1000 questions...
Cleaning HTML content...
Joining questions and answers...
Creating DPO dataset...
Created 971 DPO examples
Splitting dataset...
Converting to conversational format...
Train samples: 776
Validation samples: 97
Test samples: 98
Preprocessing completed successfully
```

#### Step 6: Log Hyperparameters For Experiments Tracking (Manual)

Instead of running computationally expensive Optuna optimization, we manually log all hyperparameters for experiment tracking:

```bash
# Run hyperparameter logging
dvc repro optimize

# Or run directly
python scripts/run_optimization.py
```

**What the optimization script does now (manual logging):**

1. Logs all configuration parameters from `config/config.yaml`.
2. Logs the search space from `config/params.yaml`.
3. Logs current training parameters.
4. Saves everything to `metrics/optimization_results.json`.
5. Creates an MLflow run with all parameters logged.
6. Does **NOT** perform actual optimization or model training.

**Expected output:**

```
============================================================
HYPERPARAMETER LOGGING (No Optimization)
============================================================

[Logging Configuration Parameters]
  config.project.name: llmops-rag-pipeline
  config.project.version: 1.0.0
  ...

[Logging Search Space Parameters]
  learning_rate: min=1e-06, max=0.0001, type=loguniform
  beta: min=0.01, max=0.5, type=uniform
  lora_r: min=4, max=32, type=int
  ...

[Logging Current Training Parameters]
  current.learning_rate: 5e-05
  current.beta: 0.1
  current.lora_r: 16
  ...

[Parameters saved to metrics/optimization_results.json]

============================================================
PARAMETER LOGGING COMPLETED
============================================================
```

#### Step 7: Train Model

```bash
# Run training pipeline
dvc repro train

# Or run directly
python scripts/run_pipeline.py
```

**What training does:**

1. Loads processed datasets from `data/processed/`.
2. Loads Qwen3-0.6B model with 4-bit quantization (QLoRA).
3. Applies LoRA adapters to specific modules.
4. Trains with DPO loss for preference alignment.
5. Saves fine-tuned model to `models/qwen-dpo-final/`.
6. Logs all metrics to MLflow.

**Key training parameters (configurable in `config/config.yaml`):**

- `num_train_epochs`: 30
- `learning_rate`: 5e-5
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 8
- `beta`: 0.1 (DPO temperature)
- `max_length`: 1024

**Training progress monitoring:**

```bash
# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi

# View MLflow logs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

#### Step 8: Evaluate Model

```bash
# Run evaluation
dvc repro evaluate

# Or run directly
python scripts/run_evaluation.py
```

**What evaluation does:**

1. Loads the fine-tuned model and test dataset.
2. Generates responses for test prompts.
3. Computes similarity metrics (BLEU, ROUGE, BERTScore).
4. Calculates perplexity.
5. Analyzes code quality (functions, imports, comments).
6. Detects programming languages in generated code.
7. Saves metrics to `metrics/evaluation_metrics.json`.
8. Creates a metrics comparison plot.

**Expected output:**

```
============================================================
EVALUATION RESULTS
============================================================
bleu: 0.0303
rouge1: 0.2324
rouge2: 0.0447
rougeL: 0.1218
bertscore_f1: 0.7956
perplexity: 2.8343
has_thinking_ratio: 0.5
has_solution_ratio: 0.4
has_explanation_ratio: 0.5
code_presence_ratio: 0.3
avg_comment_ratio: 0.4
has_imports_ratio: 0.2000
has_functions_ratio: 0.4000
avg_function_count: 0.4000
avg_line_count: 36.0000
valid_samples_ratio: 1.0000

Language Distribution:
  no_code: 1

============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================
```

#### Step 9: View MLflow Dashboard

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Open in browser
# http://localhost:5000
```

**What you can see in MLflow:**

- **Parameters:** All hyperparameters used in training.
- **Metrics:** Loss, rewards/accuracies, BLEU, ROUGE, perplexity, etc.
- **Artifacts:** Model files, evaluation JSON, plots.
- **Tags:** Run name, stage, version.
- **Search:** Filter runs by parameters or metrics.
- **Compare:** Compare multiple runs side-by-side.

---

## 8. Docker Deployment

### Build and Run Locally

```bash
# Build Docker image
docker build -t llmops-rag-model:latest .

# Run with Docker Compose
docker-compose up inference

# Or run directly
docker run --gpus all -p 8000:8000 llmops-rag-model:latest
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate response
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to reverse a list in Python?",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50
  }'
```

**Expected response:**

```json
{
  "generated_text": "To reverse a list in Python, you can use the reverse() method or slicing...",
  "latency_ms": 412
}
```

### Stop Containers

```bash
# Stop all services
docker-compose down

# Remove volumes
docker-compose down -v
```

### Docker Commands Reference

| Command | Description |
|---|---|
| `docker build -t llmops-rag-model .` | Build image |
| `docker run --gpus all -p 8000:8000 llmops-rag-model` | Run container with GPU |
| `docker ps` | List running containers |
| `docker stop <container_id>` | Stop container |
| `docker rm <container_id>` | Remove container |
| `docker rmi llmops-rag-model` | Remove image |

---

## 9. AWS Deployment

### Setup AWS Requirements

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, region (us-east-1)

# Create IAM role for Lambda (or use existing)
# Role ARN needed for deployment
```

### Deploy with Python Script

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
export LAMBDA_ROLE_ARN="arn:aws:iam::account-id:role/lambda-execution-role"

# Run deployment script
python deployment/aws_deploy.py
```

### Deploy with Terraform

```bash
# Navigate to terraform directory
cd deployment/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply deployment
terraform apply -auto-approve

# Get outputs
terraform output

# Destroy resources when done
terraform destroy
```

### API Gateway Endpoint

After deployment, you'll receive an API Gateway URL:

```
https://{api-id}.execute-api.{region}.amazonaws.com/prod/generate
```

### Production Recommendation

- **Lambda:** Suitable for demos and low-traffic endpoints. Cold starts and CPU-only inference will be slow for a 0.6B model.
- **ECS/EKS with GPU:** Recommended for production. Use ECR for images, ECS Fargate or EC2 GPU instances for serving, and an Application Load Balancer for traffic.
- **SageMaker Endpoint:** Recommended if you want managed scaling and monitoring.

---

## 10. API Usage

### Swagger Documentation

Once deployed, access the interactive API documentation at:

- **Local:** `http://localhost:8000/docs`
- **AWS:** `https://{api-id}.execute-api.{region}.amazonaws.com/prod/docs`

### Health Endpoint

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "model": "qwen-dpo-final",
  "device": "cuda"
}
```

### Generation Endpoint

```
POST /generate
```

Request Body:

```json
{
  "prompt": "How to reverse a list in Python?",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 50
}
```

Response:

```json
{
  "generated_text": "You can reverse a list in Python using slicing: my_list[::-1] ...",
  "latency_ms": 412
}
```

### Python Client Example

```python
import requests

url = "http://localhost:8000/generate"
payload = {
    "prompt": "How to reverse a list in Python?",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## 11. Testing

### Run All Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Test data preprocessing
pytest tests/test_data.py -v

# Test model loading and generation
pytest tests/test_model.py -v

# Test API endpoints
pytest tests/test_api.py -v
```

### Test with Sample Data

```bash
# Run a quick test with minimal data
python -c "
from src.data.preprocessing import clean_html
print(clean_html('<p>Hello <b>World</b></p>'))
# Expected: 'Hello World'
"
```

---

## 12. Environment Variables

Create a `.env` file for persistent configuration:

```bash
# DagsHub Configuration
DAGSHUB_USER=your-username
DAGSHUB_TOKEN=your-token
MLFLOW_TRACKING_URI=https://dagshub.com/your-username/your-repo.mlflow

# HuggingFace Configuration
HF_TOKEN=hf_your-token

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
LAMBDA_ROLE_ARN=arn:aws:iam::account-id:role/lambda-execution-role

# Local Paths
MODEL_PATH=./models/qwen-dpo-final
DATA_PATH=./data/raw
OUTPUT_PATH=./models

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME=llmops-rag-pipeline
```

Load environment variables:

```bash
source .env
# Or use python-dotenv
```

---

## 13. Monitoring and Logging

### MLflow Metrics Tracked

**Training Metrics:**

- `loss`: Cross-entropy loss
- `rewards/accuracies`: Preference accuracy
- `rewards/margins`: Margin between chosen and rejected
- `grad_norm`: Gradient norm
- `learning_rate`: Current learning rate
- `entropy`: Model entropy

**Evaluation Metrics:**

- `bleu`: BLEU score for generated text
- `rouge1`, `rouge2`, `rougeL`: ROUGE scores
- `bertscore_f1`: BERTScore F1
- `perplexity`: Model perplexity

**Code Quality Metrics:**

- `has_functions_ratio`: Percentage with functions
- `has_imports_ratio`: Percentage with imports
- `has_comments_ratio`: Percentage with comments
- `avg_function_count`: Average number of functions
- `avg_line_count`: Average lines of code

### View MLflow Dashboard

```bash
# Local tracking
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# DagsHub tracking (if configured)
# Visit: https://dagshub.com/{username}/{repo}.mlflow
```

### View DVC Pipeline Status

```bash
# Show pipeline DAG
dvc dag

# Show pipeline status
dvc status

# Show metrics
dvc metrics show

# Show plots
dvc plots show
```

### Production Monitoring Recommendations

- **API Latency:** Track p50/p95/p99 latency for `/generate`.
- **Error Rates:** Monitor 4xx/5xx responses via API Gateway or ALB logs.
- **GPU Utilization:** Use CloudWatch or Prometheus with NVIDIA DCGM exporter for GPU-backed serving.
- **Model Drift:** Periodically re-run evaluation on fresh prompts and compare metrics against the MLflow baseline.
- **Cost:** Track GPU hours, Lambda invocations, and ECR storage.

---

## 14. Troubleshooting

### DagsHub Authentication Issues

```bash
# Reconfigure DVC remote
dvc remote remove origin
dvc remote add origin https://dagshub.com/${DAGSHUB_USER}/${DAGSHUB_REPO}.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user ${DAGSHUB_USER}
dvc remote modify origin --local password ${DAGSHUB_TOKEN}

# Test connection
dvc pull -r origin
```

### MLflow Permission Issues

```bash
# Use local tracking instead of remote
export MLFLOW_TRACKING_URI="file:./mlruns"

# Or create a new experiment
mlflow experiments create -n llmops-rag-pipeline
```

### Out of Memory (OOM)

```yaml
# Reduce batch size in config/config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Reduce max length
max_length: 512

# Enable gradient checkpointing (already enabled)
gradient_checkpointing: true
```

### Data Loading Issues

```bash
# Verify CSV files exist
ls -la data/raw/

# Check CSV encoding
file data/raw/Questions.csv

# Re-run preprocessing with smaller sample
# In config/config.yaml, set:
sample_size: 100  # instead of 1000
```

### Docker Issues

```bash
# Check if Docker is installed
docker --version

# Verify GPU is available to Docker
docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Install NVIDIA Container Toolkit if GPU is not detected
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### AWS Lambda Timeouts

- Lambda has a 15-minute maximum timeout and limited memory.
- For a 0.6B model, cold starts and CPU-only inference will be slow.
- **Recommended:** move to ECS with GPU or SageMaker endpoint for production.

---

## 15. Engineering Considerations and Limitations

### 15.1 Ethical and Operational Considerations

- Stack Overflow data is used under its original license; users must verify terms before redistribution.
- Generated code may contain insecure, biased, or outdated patterns; human review is required before any production use.
- Automatic metrics are proxies and should not replace human evaluation.
- The model should not be deployed in high-stakes settings without further validation and human-in-the-loop processes.

### 15.2 Technical Limitations

- BLEU, ROUGE, and BERTScore may not fully reflect code correctness; execution-based evaluation is not implemented.
- 4-bit quantization may affect generation quality relative to a full-precision baseline.
- Hyperparameters are manually logged rather than optimized; the ablation surface is documented but not automatically swept.
- AWS Lambda is suitable for demos; GPU-backed serving (ECS, EKS, or SageMaker) is recommended for production.
- Stack Overflow data may contain biases and outdated solutions that the model will inherit.

---

## 16. Conclusion

This repository demonstrates that a small, preference-aligned code model (Qwen3-0.6B) fine-tuned with DPO + QLoRA can be taken from raw Stack Overflow data through training, evaluation, packaging, and deployment using a reproducible LLMOps pipeline. Key engineering outcomes:

- **Preference alignment is measurable.** MLflow-tracked `rewards/accuracies` and `rewards/margins` provide direct signals of DPO training progress, complementing classical generation metrics.
- **Code-quality proxies complement similarity metrics.** BLEU (0.0303), ROUGE-1 (0.2324), BERTScore F1 (0.7956), and perplexity (2.8343) on the held-out test set indicate that preference-aligned generation is coherent and semantically close to reference answers, while code-structure signals (function count, line count, import presence) correlate with perceived usefulness.
- **4-bit QLoRA is practical.** Fine-tuning on a single GPU with at least 8GB VRAM is feasible, with no requirement for multi-GPU infrastructure.
- **Deployment is reproducible.** MLflow on DagsHub, DVC versioning, Docker containerisation, and AWS deployment (ECR, Lambda/API Gateway, Terraform) form a complete, auditable production path.
- **Ablation surface is documented.** LoRA rank, DPO beta, and max sequence length are surfaced as tunable engineering knobs for future experiments.

The pipeline is designed as an engineering artefact: every experiment is tracked, every artefact is versioned, and every stage is independently testable. It supports defensive ablation of preference alignment, generation quality, and deployment overhead — and it is intended to be forked, extended, and reused for downstream LLMOps work.

---

## 17. Contributing

Contributions are welcome for production hardening, evaluation metrics, ablation support, and documentation. Please open an issue or pull request with a clear description, and ensure tests pass before submitting.

**Note:** This pipeline is designed for production and educational purposes. For real deployments, ensure proper security, monitoring, scaling, access control, and compliance with the licenses of the data and base model.

---

## 18. References

1. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS.
2. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
3. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
4. Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS.
5. Christiano, P., et al. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS.
6. Zhang, T., et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR.
7. Papineni, K., et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL.
8. Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
9. Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.
10. Austin, J., et al. (2021). *Program Synthesis with Large Language Models.* arXiv:2108.07732.
11. Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
12. Garg, S., et al. (2021). *MLOps: Continuous Delivery and Automation Pipelines in Machine Learning.* arXiv:2105.06417.
13. Kreuzberger, D., et al. (2023). *Machine Learning Operations (MLOps): Overview, Definition, and Architecture.* IEEE Access.
14. Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108.
15. Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971.
16. Qwen Team (2024). *Qwen Technical Report.* Alibaba Group.

Additional resources referenced by the original project:

- Hugging Face `transformers`, `peft`, `trl`, and `bitsandbytes` for quantized fine-tuning.
- MLflow and DagsHub documentation for experiment tracking.
- DVC documentation for data and model versioning.
- FastAPI, Docker, Terraform, and AWS documentation for deployment.

---

## 19. License

MIT License - see `LICENSE` file for details.

Provided for production and educational purposes. Redistribution of the pipeline outside a production-engineering or educational context is discouraged; if you fork this repository, preserve the [§15 Engineering Considerations and Limitations](#15-engineering-considerations-and-limitations) section verbatim.

**Note:** This pipeline is designed for production and educational purposes. For real deployments, ensure proper security, monitoring, scaling, access control, and compliance with the licenses of the data and base model.

---

## About

This repository is a comprehensive LLMOps pipeline for fine-tuning LLM with Direct Preference Optimization (DPO) and QLoRA to improve code-generation quality, consisting of experiment tracking, data versioning, multi-metric evaluation, Dockerized FastAPI inference, and AWS deployment.

### Resources

- Readme
- MIT license

### Activity

- Stars: 0 stars
- Watchers: 0 watching
- Forks: 0 forks

### Repository Purpose

This repository is maintained as a reproducible production-engineering artefact. It is intended to support:

- **Preference-alignment engineering** on small open models with QLoRA + DPO.
- **MLOps toolchain evaluation** using MLflow, DagsHub, and DVC.
- **Reproducible production deployment** of small code-generation services via Docker and AWS.
- **Educational use** as a working codebase that demonstrates full-lifecycle LLM engineering.

It demonstrates full-lifecycle systems thinking — data versioning, preference-pair construction, fine-tuning, evaluation, deployment, and monitoring — with direct applicability to LLMOps engineering and production ML engineering roles.
