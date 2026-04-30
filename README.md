# Enterprise-Scale NLP with Hugging Face & Amazon SageMaker

![Hugging Face & Amazon SageMaker](./imgs/cover.png)

This repository contains a comprehensive set of resources and code demonstrations for building, training, deploying, scaling, and monitoring enterprise-scale Natural Language Processing (NLP) models using Hugging Face Transformers and Amazon SageMaker.

This repository serves as an extensive portfolio of advanced Machine Learning Engineering and MLOps practices on AWS.

## 📌 Table of Contents
- [Overview](#overview)
- [Architecture & Workshops](#architecture--workshops)
  - [1. Getting Started with Amazon SageMaker](#1-getting-started-with-amazon-sagemaker)
  - [2. Going to Production](#2-going-to-production)
  - [3. MLOps End-to-End Pipeline](#3-mlops-end-to-end-pipeline)
  - [4. Distillation and Acceleration](#4-distillation-and-acceleration)
- [Getting Started (AWS Setup)](#getting-started-aws-setup)
- [License](#license)

## 🚀 Overview

Through a strategic collaboration between Hugging Face and Amazon, it has become easier than ever to train and deploy Hugging Face Transformers using Amazon SageMaker. 
This project leverages **Hugging Face Deep Learning Containers (DLCs)** and the **Hugging Face Inference Toolkit** for SageMaker to enable zero-code deployments of cutting-edge models.

### Key Technologies Demonstrated:
- **Hugging Face Transformers**: `transformers`, `datasets`, `tokenizers`
- **Amazon SageMaker**: Training Jobs, Real-time Endpoints, Batch Transform, Spot Instances, Pipelines (MLOps)
- **AWS Inferentia**: Accelerating deep learning inference
- **Optimization**: Knowledge Distillation, AWS Neuron

---

## 🏗 Architecture & Workshops

The repository is structured into four core areas, moving from initial experimentation to advanced MLOps and hardware optimization.

### 1. Getting Started with Amazon SageMaker
**Folder**: [`workshop_1_getting_started_with_amazon_sagemaker/`](./workshop_1_getting_started_with_amazon_sagemaker/)

Learn how to use Amazon SageMaker to train a Hugging Face Transformer model and deploy it.
- **Data Preparation**: Prepare and upload a test dataset to S3.
- **Fine-Tuning**: Prepare a fine-tuning script for SageMaker Training jobs.
- **Training**: Launch a training job, utilize Spot Instances for cost savings, and store the trained model into S3.
- **Deployment**: Deploy the model to a real-time SageMaker endpoint.

### 2. Going to Production
**Folder**: [`workshop_2_going_production/`](./workshop_2_going_production/)

Deploy, scale, and monitor Hugging Face Transformer models for production workloads.
- **Batch Inference**: Run predictions on large datasets using SageMaker Batch Transform.
- **Zero-Code Deployment**: Deploy a model directly from the Hugging Face Hub to SageMaker.
- **Auto-Scaling**: Configure dynamic autoscaling for deployed models based on traffic.
- **Monitoring**: Track model performance (avg. request time) and set up CloudWatch alarms.

### 3. MLOps End-to-End Pipeline
**Folder**: [`workshop_3_mlops/`](./workshop_3_mlops/)

Build an End-to-End MLOps Pipeline for Hugging Face Transformers from training to production.
- **SageMaker Pipelines**: Automate the entire ML lifecycle.
- **Data Processing**: Preprocess data and upload it to S3 as part of the pipeline.
- **Model Training**: Fine-tune a model conditionally.
- **Evaluation & Deployment**: Evaluate the model against a test set and conditionally deploy it only if it exceeds a performance threshold.

### 4. Distillation and Acceleration
**Folder**: [`workshop_4_distillation_and_acceleration/`](./workshop_4_distillation_and_acceleration/)

Optimize models for ultra-low latency inference using Knowledge Distillation and AWS Inferentia.
- **Knowledge Distillation**: Compress a large model (BERT-large) to a smaller student model (MiniLM).
- **AWS Neuron**: Compile a Hugging Face Transformer model for AWS Inferentia hardware.
- **High-Performance Deployment**: Deploy the distilled and optimized model for a 20x latency improvement.

---

## ⚙️ Getting Started (AWS Setup)

To run the notebooks and code in this repository, you will need access to an AWS Account with permissions to use Amazon SageMaker.

### Prerequisites

1. **AWS Account**: You must have an active AWS account.
2. **IAM Roles**: Create an IAM Execution Role for SageMaker with the following policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (Or restrict to the specific S3 buckets you will use)

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/huggingface-sagemaker-workshop-series.git
   cd huggingface-sagemaker-workshop-series
   ```

2. **Install Dependencies** (Local Environment / SageMaker Studio Lab):
   Create a virtual environment and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Amazon SageMaker Notebook Instance**:
   Alternatively, you can run these notebooks directly on a SageMaker Notebook Instance:
   - Go to the Amazon SageMaker Console.
   - Click **Notebook Instances** -> **Create notebook instance**.
   - Attach your IAM Role and ensure the volume size is large enough (e.g., 50GB+).
   - Under **Git repositories**, you can provide the link to this repository to have it cloned automatically upon creation.

4. **Run the Notebooks**:
   Open JupyterLab, navigate to the desired workshop folder, select the `conda_pytorch_p39` (or similar PyTorch kernel), and start running the cells!

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
