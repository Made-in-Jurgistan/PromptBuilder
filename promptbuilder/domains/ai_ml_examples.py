"""
ai_ml_examples.py

This module contains all AI/ML domain code examples for PromptBuilder, strictly separated from domain metadata/configuration. Each example is a dictionary entry with task, description, language, code, and explanation fields. This file is intended for code retrieval, documentation, and demonstration purposes only.
"""

from typing import List, Dict, Any

ai_ml_code_examples: List[Dict[str, Any]] = [
    {
        "task": "Train a Transformer-based Text Classifier with PyTorch",
        "description": "Demonstrates how to fine-tune a Hugging Face Transformer model for text classification using PyTorch and the Transformers library. Useful for sentiment analysis, topic classification, or intent detection.",
        "language": "Python",
        "code": """
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset('imdb')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

tokenized = dataset.map(tokenize, batched=True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized['test'].shuffle(seed=42).select(range(500)),
)

trainer.train()
""",
        "explanation": "This example shows how to fine-tune a pre-trained DistilBERT model for binary text classification using the Hugging Face Transformers and Datasets libraries. It covers dataset loading, tokenization, model instantiation, and training with the Trainer API. This approach is production-ready, supports GPU acceleration, and can be adapted for other text classification tasks by changing the dataset or model. Best practices include using a validation set, shuffling data, and leveraging pre-trained models for transfer learning."
    },
    {
        "task": "Distributed Training of a Large Language Model with DeepSpeed",
        "description": "Illustrates how to set up distributed training for a large language model using DeepSpeed and PyTorch. Essential for scaling LLM training across multiple GPUs or nodes.",
        "language": "Python",
        "code": """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

model_name = 'gpt2-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# DeepSpeed config (usually in a JSON file)
deep_speed_config = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 2}
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=deep_speed_config
)

input_ids = tokenizer("DeepSpeed distributed training!", return_tensors="pt").input_ids
outputs = model_engine(input_ids)
print(outputs)
""",
        "explanation": "This code demonstrates distributed training of a GPT-2 model using DeepSpeed, a popular library for efficient large-scale model training. The configuration enables mixed-precision (fp16) and ZeRO optimization for memory efficiency. DeepSpeed's initialize function wraps the model for distributed execution. This setup is essential for training LLMs that exceed single-GPU memory, and is widely used in production and research."
    },
    {
        "task": "Build a Retrieval-Augmented Generation (RAG) Pipeline with LangChain and FAISS",
        "description": "Shows how to create a RAG pipeline using LangChain to combine LLMs with vector search for document question answering. Useful for enterprise search, chatbots, and knowledge assistants.",
        "language": "Python",
        "code": """
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load documents and create embeddings
documents = ["LangChain enables RAG workflows.", "FAISS is a vector database."]
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)

# Set up LLM and RAG chain
llm = OpenAI(model_name="gpt-3.5-turbo")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

result = qa_chain.run("What is LangChain?")
print(result)
""",
        "explanation": "This example builds a simple RAG pipeline using LangChain and FAISS. Documents are embedded and indexed for vector search. The RetrievalQA chain combines an LLM with a retriever to answer questions using both retrieval and generation. This pattern is foundational for modern enterprise search, chatbots, and knowledge assistants, and is a best practice for grounding LLM outputs in external data."
    },
    {
        "task": "Serve a Machine Learning Model with FastAPI",
        "description": "Demonstrates how to deploy a trained ML model as a REST API using FastAPI. Suitable for production inference endpoints and microservices.",
        "language": "Python",
        "code": """
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.joblib')

class Input(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(input: Input):
    data = [[input.feature1, input.feature2]]
    prediction = model.predict(data)
    return {"prediction": prediction[0]}
""",
        "explanation": "This code shows how to serve a trained ML model as a REST API using FastAPI, a modern Python web framework. The model is loaded with joblib, and a POST endpoint receives input features and returns predictions. This approach is widely used for production inference, supports async operation, and can be containerized for scalable deployment."
    },
    {
        "task": "Track Experiments and Visualize Metrics with Weights & Biases",
        "description": "Illustrates how to use Weights & Biases (wandb) for experiment tracking and metric visualization during model training. Essential for reproducibility and collaboration in ML projects.",
        "language": "Python",
        "code": """
import wandb
import numpy as np

wandb.init(project="ml-demo")
for epoch in range(5):
    acc = np.random.rand()
    loss = np.random.rand()
    wandb.log({"accuracy": acc, "loss": loss, "epoch": epoch})
wandb.finish()
""",
        "explanation": "This example demonstrates the use of Weights & Biases for logging metrics during training. wandb.init starts a new run, and wandb.log records metrics at each epoch. This enables experiment tracking, visualization, and collaboration, which are best practices for modern ML workflows."
    },
    {
        "task": "Orchestrate ML Pipelines with Apache Airflow",
        "description": "Shows how to define and schedule a simple ML pipeline using Apache Airflow. Useful for automating ETL, training, and deployment workflows in production.",
        "language": "Python",
        "code": """
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def preprocess():
    print("Preprocessing data...")

def train():
    print("Training model...")

def deploy():
    print("Deploying model...")

default_args = {"start_date": datetime(2024, 1, 1)}
dag = DAG("ml_pipeline", schedule_interval="@daily", default_args=default_args, catchup=False)

preprocess_task = PythonOperator(task_id="preprocess", python_callable=preprocess, dag=dag)
train_task = PythonOperator(task_id="train", python_callable=train, dag=dag)
deploy_task = PythonOperator(task_id="deploy", python_callable=deploy, dag=dag)

preprocess_task >> train_task >> deploy_task
""",
        "explanation": "This code defines a simple ML pipeline in Apache Airflow, with tasks for preprocessing, training, and deployment. Each task is a Python function, and dependencies are set using the >> operator. Airflow is a production-grade workflow orchestrator, and this pattern is widely used for automating ML pipelines."
    },
    {
        "task": "Export and Optimize Models with ONNX",
        "description": "Demonstrates how to export a trained PyTorch model to ONNX format and run inference with ONNX Runtime. Useful for cross-framework deployment and hardware optimization.",
        "language": "Python",
        "code": """
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
dummy_input = torch.randn(1, 2)
torch.onnx.export(model, dummy_input, "model.onnx")
sess = ort.InferenceSession("model.onnx")
outputs = sess.run(None, {sess.get_inputs()[0].name: dummy_input.numpy()})
print(outputs)
""",
        "explanation": "This example shows how to export a PyTorch model to ONNX format and run inference using ONNX Runtime. ONNX enables cross-framework model deployment and hardware optimization. This workflow is common in production for deploying models to diverse environments, including cloud and edge devices."
    },
    {
        "task": "Monitor Data Drift with Evidently AI",
        "description": "Shows how to use Evidently AI to detect data drift between training and production datasets. Essential for monitoring model health and triggering retraining in production.",
        "language": "Python",
        "code": """
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

train = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
prod = pd.DataFrame({"feature": [1, 2, 2, 3, 100]})

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train, current_data=prod)
report.show()
""",
        "explanation": "This code uses Evidently AI to compare a training dataset and a production dataset for data drift. The Report object runs a preset drift analysis and displays the results. Monitoring for drift is a best practice for production ML systems to ensure ongoing model reliability."
    }
]

# NOTE: This file contains only code examples and their explanations. No domain metadata, mappings, or configuration is present here.
