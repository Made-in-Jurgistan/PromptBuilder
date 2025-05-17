"""
AI/ML Domain Module for PromptBuilder.

This module contains the comprehensive technology mapping for the AI/ML domain
within the PromptBuilder system, including languages, frameworks, libraries, 
and domain-specific concepts.

Author: Made in Jurgistan
Version: 2.0.0
License: MIT
"""

from typing import Dict, Any


def get_ai_ml_mapping() -> Dict[str, Dict[str, Any]]:
    """
    Get AI/ML technology mapping.
    
    Returns:
        Dict[str, Dict[str, Any]]: Comprehensive AI/ML domain mapping adhering 
        to the standardized template
    """
    return {
        # DOMAIN METADATA
        "name": "AI/ML",
        "description": "Development and application of artificial intelligence and machine learning systems for predictive modeling, natural language processing, computer vision, and other intelligent systems. This domain encompasses both theoretical foundations and practical implementations that enable machines to learn from data and make decisions.",
        
        # LANGUAGES (EXACTLY 6 entries)
        "languages": [
            {
                "name": "Python",
                "version": "3.11",
                "description": "Primary language for AI/ML development with extensive library support that enables rapid prototyping and deployment. Python's simple syntax and powerful ecosystem make it the de facto standard for data science, machine learning research, and production AI systems.",
                "popularity": "high",
                "typical_uses": ["Model development", "Data preprocessing", "Research implementation", "Production AI systems"]
            },
            {
                "name": "R",
                "version": "4.3",
                "description": "Statistical computing language popular in data science and analysis with specialized packages for statistical modeling. R excels in exploratory data analysis, visualization, and statistical inference with a rich ecosystem of packages specifically designed for statistical computing.",
                "popularity": "medium",
                "typical_uses": ["Statistical analysis", "Data visualization", "Research", "Specialized statistical models"]
            },
            {
                "name": "Julia",
                "version": "1.9",
                "description": "High-performance language for technical computing with Python-like syntax but C-like performance. Julia addresses the two-language problem by providing both ease of use and performance, making it ideal for computationally intensive ML applications and numerical simulations.",
                "popularity": "low",
                "typical_uses": ["High-performance computing", "Scientific computing", "Mathematical optimization", "Numerical simulation"]
            },
            {
                "name": "Scala",
                "version": "3.3",
                "description": "JVM language used with Apache Spark for distributed computing and large-scale data processing. Scala combines object-oriented and functional programming paradigms, providing type safety and conciseness for building robust distributed ML pipelines.",
                "popularity": "medium",
                "typical_uses": ["Big data processing", "Distributed ML pipelines", "Production systems", "Stream processing"]
            },
            {
                "name": "Java",
                "version": "21",
                "description": "Enterprise language used for production ML systems deployment where stability and integration with existing systems are paramount. Java's strong typing, extensive tooling, and enterprise support make it suitable for building mission-critical AI applications with strict reliability requirements.",
                "popularity": "medium",
                "typical_uses": ["Production ML systems", "Enterprise integration", "Model serving", "Legacy system augmentation"]
            },
            {
                "name": "C++",
                "version": "20",
                "description": "High-performance language used for ML library cores and computationally intensive algorithms. C++ provides low-level memory management and optimization capabilities essential for building efficient ML infrastructure components and hardware-accelerated implementations of ML algorithms.",
                "popularity": "medium",
                "typical_uses": ["ML library development", "Performance-critical algorithms", "Embedded AI", "Hardware acceleration"]
            }
        ],
        
        # FRAMEWORKS (EXACTLY 8 entries)
        "frameworks": [
            {
                "name": "TensorFlow",
                "version": "2.13",
                "description": "End-to-end open source platform for machine learning with comprehensive tools for model development, training, and deployment. TensorFlow excels in production environments with robust serving capabilities, while supporting research with flexible APIs and visualization tools like TensorBoard.",
                "language": "Python/C++",
                "typical_uses": ["Deep learning", "Neural networks", "Production ML systems", "Research"],
                "learning_curve": "steep",
                "category": "Deep Learning"
            },
            {
                "name": "PyTorch",
                "version": "2.0",
                "description": "Open source machine learning framework with dynamic computational graph that allows for intuitive debugging and rapid experimentation. PyTorch has gained significant traction in research communities for its pythonic design, flexibility in model construction, and strong GPU acceleration for deep learning.",
                "language": "Python/C++",
                "typical_uses": ["Research", "Computer vision", "NLP", "Reinforcement learning"],
                "learning_curve": "moderate",
                "category": "Deep Learning"
            },
            {
                "name": "scikit-learn",
                "version": "1.3",
                "description": "Simple and efficient tools for data analysis and modeling in Python, focusing on classical machine learning algorithms. Scikit-learn provides consistent APIs, extensive documentation, and robust implementations that make it the go-to library for standard ML algorithms and preprocessing techniques.",
                "language": "Python",
                "typical_uses": ["Classical ML algorithms", "Data preprocessing", "Model evaluation", "Feature engineering"],
                "learning_curve": "gentle",
                "category": "Machine Learning"
            },
            {
                "name": "Keras",
                "version": "2.13",
                "description": "High-level neural networks API running on top of TensorFlow with a focus on user experience and rapid prototyping. Keras provides a simple, consistent interface while maintaining flexibility for advanced use cases, making deep learning accessible to both beginners and experienced practitioners.",
                "language": "Python",
                "typical_uses": ["Deep learning prototyping", "Neural network design", "Transfer learning", "Educational purposes"],
                "learning_curve": "gentle",
                "category": "Deep Learning"
            },
            {
                "name": "Hugging Face Transformers",
                "version": "4.31",
                "description": "State-of-the-art natural language processing library with pre-trained models for a wide range of NLP tasks. It provides unified APIs for working with thousands of pretrained models, simplifying the use of cutting-edge language models for tasks from text classification to generation.",
                "language": "Python",
                "typical_uses": ["NLP tasks", "Text generation", "Fine-tuning language models", "Transfer learning"],
                "learning_curve": "moderate",
                "category": "NLP"
            },
            {
                "name": "LightGBM",
                "version": "3.3",
                "description": "Gradient boosting framework optimized for efficiency and performance, particularly suitable for large-scale prediction tasks with structured data. LightGBM uses histogram-based learning and leaf-wise tree growth strategies to provide faster training speeds and better accuracy than many other boosting implementations.",
                "language": "C++/Python",
                "typical_uses": ["Structured data prediction", "Ranking", "Classification", "Large-scale learning"],
                "learning_curve": "moderate",
                "category": "Gradient Boosting"
            },
            {
                "name": "XGBoost",
                "version": "1.7",
                "description": "Optimized gradient boosting library with distributed computing support that has dominated ML competitions for structured data problems. XGBoost combines algorithmic optimizations and hardware acceleration to deliver state-of-the-art performance for tabular data tasks across a wide range of domains.",
                "language": "C++/Python",
                "typical_uses": ["Structured data ML", "Competitions", "Production systems", "Feature importance analysis"],
                "learning_curve": "moderate",
                "category": "Gradient Boosting"
            },
            {
                "name": "fastai",
                "version": "2.7",
                "description": "Deep learning library built on PyTorch focusing on simplicity and best practices that accelerates the development cycle for common ML tasks. Fastai provides high-level APIs that implement proven techniques for computer vision, NLP, and tabular data, allowing practitioners to quickly achieve state-of-the-art results.",
                "language": "Python",
                "typical_uses": ["Computer vision", "NLP", "Tabular data", "Transfer learning"],
                "learning_curve": "gentle",
                "category": "Deep Learning"
            }
        ],
        
        # TECHNOLOGIES (EXACTLY 8 entries)
        "technologies": [
            {
                "name": "NVIDIA CUDA",
                "version": "12.0",
                "description": "Parallel computing platform and API for GPU acceleration that has revolutionized deep learning by enabling massive parallelism. CUDA provides the foundation for neural network training acceleration, reducing training times from weeks to hours and enabling the development of increasingly complex models.",
                "category": "Hardware Acceleration",
                "maturity": "mature"
            },
            {
                "name": "TPU (Tensor Processing Unit)",
                "version": "v4",
                "description": "Google's custom-developed ASIC for machine learning acceleration, specifically optimized for TensorFlow operations. TPUs provide specialized matrix multiplication units and high-bandwidth memory, delivering exceptional performance-per-watt for large-scale neural network training and inference.",
                "category": "Hardware Acceleration",
                "maturity": "established"
            },
            {
                "name": "MLflow",
                "version": "2.4",
                "description": "Platform for managing the ML lifecycle including experimentation, reproducibility, deployment, and central model registry. MLflow solves key challenges in ML development by tracking experiments, packaging code and dependencies, and providing standardized deployment workflows across diverse environments.",
                "category": "MLOps",
                "maturity": "established"
            },
            {
                "name": "Kubeflow",
                "version": "1.7",
                "description": "Machine learning toolkit for Kubernetes that enables scalable, portable, and reproducible ML workflows. Kubeflow provides a Kubernetes-native platform for deploying, monitoring, and managing ML systems in production, with components for pipeline orchestration, notebook management, and model serving.",
                "category": "MLOps",
                "maturity": "established"
            },
            {
                "name": "Ray",
                "version": "2.5",
                "description": "Framework for distributed computing and scaling Python applications across clusters of machines. Ray provides primitives for parallel and distributed execution that enable ML workloads to scale seamlessly from laptops to clusters, with specialized libraries for hyperparameter tuning and reinforcement learning.",
                "category": "Distributed Computing",
                "maturity": "established"
            },
            {
                "name": "ONNX",
                "version": "1.14",
                "description": "Open format for machine learning models with cross-framework compatibility, enabling model portability between different tools. ONNX addresses framework lock-in by providing a common representation for neural network models that can be exchanged between frameworks and optimized for different hardware targets.",
                "category": "Model Interoperability",
                "maturity": "established"
            },
            {
                "name": "TensorRT",
                "version": "8.6",
                "description": "NVIDIA's platform for high-performance deep learning inference optimization and deployment. TensorRT provides model optimization, quantization, and compilation capabilities that dramatically accelerate inference on NVIDIA GPUs, enabling real-time applications of complex neural networks.",
                "category": "Inference Optimization",
                "maturity": "established"
            },
            {
                "name": "Apache Airflow",
                "version": "2.6",
                "description": "Platform for orchestrating and scheduling workflows and data pipelines critical for production ML systems. Airflow enables the programmatic authoring, scheduling, and monitoring of complex data processing and model training pipelines through directed acyclic graphs of tasks.",
                "category": "MLOps",
                "maturity": "established"
            }
        ],
        
        # CONCEPTS (EXACTLY 50 entries)
        "concepts": [
            "Activation Functions",
            "Active Learning",
            "Attention Mechanisms",
            "Autoencoder",
            "AutoML",
            "Backpropagation",
            "Batch Normalization",
            "Bias-Variance Tradeoff",
            "Convolutional Neural Networks (CNN)",
            "Cross-Validation",
            "Data Augmentation",
            "Deep Learning",
            "Dimensionality Reduction",
            "Embeddings",
            "Ensemble Methods",
            "Explainable AI (XAI)",
            "Feature Engineering",
            "Feature Selection",
            "Few-Shot Learning",
            "Foundation Models",
            "Federated Learning",
            "Generative Adversarial Networks (GANs)",
            "Gradient Descent",
            "Hyperparameter Tuning",
            "Imbalanced Learning",
            "Knowledge Distillation",
            "Large Language Models (LLMs)",
            "Model Compression",
            "Model Drift",
            "Model Interpretability",
            "Model Quantization",
            "MLOps",
            "Natural Language Processing (NLP)",
            "Neural Architecture Search",
            "Neural Networks",
            "Online Learning",
            "Overfitting and Underfitting",
            "Precision and Recall",
            "Reinforcement Learning",
            "Recurrent Neural Networks (RNN)",
            "Regularization",
            "Responsible AI",
            "ROC Curve and AUC",
            "Self-Supervised Learning",
            "Semi-Supervised Learning",
            "Supervised Learning",
            "Transfer Learning",
            "Transformers",
            "Unsupervised Learning",
            "Zero-Shot Learning"
        ],
        
        # BEST PRACTICES (EXACTLY 10 entries)
        "best_practices": [
            {
                "name": "Data Validation and Cleaning",
                "description": "Systematically validating and cleaning data before model training to identify and address issues such as missing values, outliers, and inconsistencies. This practice establishes a foundation of data quality that directly impacts model performance, generalization ability, and the reliability of predictions in production.",
                "benefits": ["Improved model performance", "Reduced errors", "Reproducible results", "Better generalization"],
                "related_concepts": ["Feature Engineering", "Data Augmentation", "MLOps", "Model Drift"]
            },
            {
                "name": "Experiment Tracking",
                "description": "Systematically recording all experimental parameters, datasets, results, and model artifacts to ensure reproducibility and facilitate analysis. Comprehensive tracking enables comparison between approaches, facilitates collaboration among team members, and provides an audit trail for model development decisions and performance improvements.",
                "benefits": ["Reproducibility", "Collaboration efficiency", "Model governance", "Faster iteration"],
                "related_concepts": ["MLOps", "Hyperparameter Tuning", "Cross-Validation", "Model Drift"]
            },
            {
                "name": "Cross-Validation",
                "description": "Evaluating models on multiple data splits to ensure generalization capabilities and stability across different subsets of data. This technique provides more reliable performance estimates than single train-test splits, helps detect overfitting, and enables more confident model selection based on consistent performance metrics.",
                "benefits": ["Reliable performance estimates", "Reduced overfitting", "Better model selection", "Robustness"],
                "related_concepts": ["Overfitting and Underfitting", "Bias-Variance Tradeoff", "Model Evaluation", "Hyperparameter Tuning"]
            },
            {
                "name": "Model Versioning",
                "description": "Maintaining systematic versioning of models, data, and code to ensure reproducibility, facilitate rollbacks, and enable clear progression tracking. Proper versioning creates a complete lineage for each model, linking artifacts to their training data, parameters, and code, which is essential for auditing and compliance requirements.",
                "benefits": ["Reproducibility", "Auditability", "Rollback capability", "Collaboration"],
                "related_concepts": ["MLOps", "Experiment Tracking", "Model Drift", "Responsible AI"]
            },
            {
                "name": "Feature Selection",
                "description": "Identifying and using only the most informative features for model training to improve performance, reduce overfitting, and decrease computational requirements. Strategic feature selection eliminates noise and redundancy in the input data, leading to models that are both more accurate and more interpretable.",
                "benefits": ["Improved performance", "Reduced overfitting", "Lower computational cost", "Better interpretability"],
                "related_concepts": ["Feature Engineering", "Dimensionality Reduction", "Regularization", "Model Interpretability"]
            },
            {
                "name": "Responsible AI Development",
                "description": "Developing ML systems with explicit consideration of fairness, transparency, accountability, and ethical implications throughout the entire development lifecycle. This practice ensures that AI systems deliver value while minimizing potential harms, addressing biases, and maintaining alignment with human values and societal norms.",
                "benefits": ["Reduced bias", "Legal compliance", "User trust", "Ethical alignment"],
                "related_concepts": ["Explainable AI (XAI)", "Model Interpretability", "Fairness Metrics", "Responsible AI"]
            },
            {
                "name": "Model Monitoring",
                "description": "Continuously monitoring model performance and health in production to detect drift, degradation, or unexpected behavior as early as possible. Effective monitoring includes tracking input data distributions, prediction distributions, model accuracy, and business KPIs to ensure models remain reliable and effective over time.",
                "benefits": ["Early drift detection", "Performance maintenance", "Issue remediation", "SLA compliance"],
                "related_concepts": ["Model Drift", "MLOps", "Data Validation", "Online Learning"]
            },
            {
                "name": "Ensemble Learning",
                "description": "Combining multiple models to improve prediction performance, stability, and robustness beyond what individual models can achieve alone. Ensemble techniques like bagging, boosting, and stacking leverage the diversity of different models or training approaches to compensate for individual weaknesses and enhance overall system performance.",
                "benefits": ["Improved accuracy", "Reduced variance", "Better generalization", "Robustness to outliers"],
                "related_concepts": ["Ensemble Methods", "Bias-Variance Tradeoff", "Gradient Boosting", "Model Selection"]
            },
            {
                "name": "Pipeline Automation",
                "description": "Creating automated, reproducible workflows for all stages of ML development from data ingestion and preprocessing to model training, evaluation, and deployment. Automated pipelines ensure consistency, reduce manual errors, enable continuous integration and deployment, and facilitate rapid iteration on models and features.",
                "benefits": ["Reproducibility", "Efficiency gains", "Error reduction", "Faster deployment"],
                "related_concepts": ["MLOps", "CI/CD for ML", "Experiment Tracking", "Model Versioning"]
            },
            {
                "name": "Transfer Learning",
                "description": "Leveraging knowledge from pre-trained models to improve performance on new tasks with limited data or computational resources. This approach dramatically reduces training time and data requirements by starting from general representations learned from large datasets before fine-tuning on specific target tasks.",
                "benefits": ["Reduced training data needs", "Faster convergence", "Better performance", "Resource efficiency"],
                "related_concepts": ["Transfer Learning", "Foundation Models", "Few-Shot Learning", "Fine-tuning"]
            }
        ],
        
        # COMMON PROBLEMS (EXACTLY 10 entries)
        "common_problems": [
            {
                "name": "Overfitting",
                "description": "Model performs well on training data but poorly on unseen data due to learning noise or peculiarities in the training set rather than generalizable patterns. This fundamental challenge in machine learning manifests as the model essentially memorizing training examples.",
                "indicators": ["High training accuracy but low validation accuracy", "Complex model for simple problem", "Perfect training fit", "Poor generalization", "Validation loss increases while training loss decreases"],
                "causes": ["Too complex model", "Insufficient data", "Noisy features", "Data leakage", "Training too long"],
                "solutions": ["Regularization", "More training data", "Simpler model", "Cross-validation", "Early stopping"]
            },
            {
                "name": "Data Leakage",
                "description": "Training data includes information that won't be available during inference, leading to unrealistically high performance metrics during development but failure in production. This insidious problem creates a false sense of model capability and leads to unexpected performance degradation.",
                "indicators": ["Unrealistically high performance", "Performance collapse in production", "Time-based inconsistencies", "Perfect predictions on supposed unknowns", "Suspiciously good results"],
                "causes": ["Including target-derived features", "Preprocessing before splitting", "Temporal data mishandling", "Test set contamination", "Improper cross-validation setup"],
                "solutions": ["Proper train-test splits", "Time-based validation", "Careful feature engineering", "Pipeline-based preprocessing", "Holdout validation"]
            },
            {
                "name": "Training-Serving Skew",
                "description": "Difference between training and production environments causing model performance degradation when deployed. This discrepancy often arises from differences in data processing, feature engineering implementations, or distribution shifts between training and production contexts.",
                "indicators": ["Good offline metrics but poor online performance", "Unexpected production behavior", "Feature distribution shifts", "Preprocessing inconsistencies", "Input format mismatches"],
                "causes": ["Different preprocessing in training vs. serving", "Data drift", "Software implementation differences", "Environment inconsistencies", "Numerical precision variations"],
                "solutions": ["End-to-end testing", "Feature monitoring", "Unified preprocessing", "Canary deployments", "Feature store implementation"]
            },
            {
                "name": "Model Drift",
                "description": "Model performance deteriorates over time as data patterns change, whether due to shifting user behaviors, seasonal effects, or other real-world changes. This natural degradation requires ongoing attention to maintain model effectiveness in dynamic environments.",
                "indicators": ["Declining performance metrics", "Increasing error rates", "Changing prediction distributions", "Gradual deterioration", "Unexpected pattern shifts"],
                "causes": ["Changing user behavior", "Seasonal effects", "Data source changes", "World events", "Competitive marketplace changes"],
                "solutions": ["Regular retraining", "Drift detection", "Online learning", "Robust feature engineering", "Continuous monitoring"]
            },
            {
                "name": "GPU Memory Errors",
                "description": "Out of memory errors when training deep learning models on GPUs, preventing successful completion of training runs. These technical limitations require careful management of model complexity, batch sizes, and optimization techniques to enable effective training.",
                "indicators": ["CUDA out of memory errors", "Training crashes", "GPU utilization spikes", "Unexpected termination", "Resource exhaustion warnings"],
                "causes": ["Batch size too large", "Model too complex", "Memory leaks", "Inefficient data loading", "Unnecessary tensor retention"],
                "solutions": ["Gradient accumulation", "Mixed precision training", "Model parallelism", "Smaller batch size", "Model optimization"]
            },
            {
                "name": "Imbalanced Data",
                "description": "Class distribution in training data is highly skewed, leading to models biased toward majority classes while performing poorly on minority classes. This common problem in classification tasks can severely impact model fairness and utility for rare but important cases.",
                "indicators": ["High accuracy but low recall for minority class", "Model always predicts majority class", "Poor F1 score", "Misleading overall accuracy", "Class-specific performance discrepancies"],
                "causes": ["Natural class imbalance", "Sampling bias", "Rare events detection", "Biased data collection", "Domain-specific rarity"],
                "solutions": ["Resampling techniques", "Class weighting", "SMOTE", "Focal loss", "Ensemble methods"]
            },
            {
                "name": "Slow Inference",
                "description": "Model takes too long to make predictions in production, creating bottlenecks in application performance or increased operational costs. Inference latency is critical for real-time applications and can significantly impact user experience and system scalability.",
                "indicators": ["High latency", "Timeout errors", "Resource utilization spikes", "User experience degradation", "Queuing requests"],
                "causes": ["Model complexity", "Inefficient implementation", "Resource constraints", "Serial processing", "Unoptimized deployments"],
                "solutions": ["Model quantization", "Distillation", "Pruning", "Batch processing", "Hardware acceleration"]
            },
            {
                "name": "Reproducibility Issues",
                "description": "Inability to consistently reproduce model training results due to randomness, environment variations, or insufficient tracking of parameters and dependencies. Reproducibility is fundamental to scientific validity and essential for debugging and improving ML systems.",
                "indicators": ["Different results on same code/data", "Unexplainable performance variations", "Environment-dependent outcomes", "Inconsistent behavior", "Non-deterministic training"],
                "causes": ["Random seed not fixed", "Framework version differences", "Hardware variations", "Insufficient logging", "Parameter tracking gaps"],
                "solutions": ["Fixed random seeds", "Environment containerization", "Comprehensive logging", "Version pinning", "Deterministic operations"]
            },
            {
                "name": "Feature Engineering Complexity",
                "description": "Difficulty in identifying, creating, and selecting the most relevant features for a given problem, requiring significant domain expertise and experimentation. Effective feature engineering remains a crucial human-in-the-loop aspect of many ML workflows despite advances in representation learning.",
                "indicators": ["Labor-intensive preprocessing", "Inconsistent feature quality", "Brittle pipeline dependencies", "Knowledge bottlenecks", "Manual intervention requirements"],
                "causes": ["Domain complexity", "Data heterogeneity", "Lack of expertise", "Unclear feature relevance", "Over-engineered features"],
                "solutions": ["Automated feature generation", "Representation learning", "Feature importance analysis", "Domain expert collaboration", "Iterative refinement"]
            },
            {
                "name": "Cold Start Problem",
                "description": "Challenge of making accurate predictions for new users or items with little to no historical data, particularly affecting recommendation systems and personalization engines. This inherent limitation impacts user experience for newcomers and complicates the introduction of new items.",
                "indicators": ["Poor recommendations for new users", "Inability to promote new items", "Bias toward popular items", "User dissatisfaction", "Adoption barriers"],
                "causes": ["Lack of interaction history", "Dependency on collaborative filtering", "Insufficient metadata", "Popularity bias", "Limited content features"],
                "solutions": ["Hybrid recommendation approaches", "Content-based features", "Default personas", "Exploration strategies", "Transfer learning from similar users"]
            }
        ],
        
        # CODE EXAMPLES (EXACTLY 8 entries)
        "code_examples": [
            {
                "task": "Image Classification with PyTorch",
                "description": "This implementation demonstrates a comprehensive PyTorch-based convolutional neural network for image classification, featuring data augmentation, batch normalization, and a complete training pipeline.",
                "language": "Python",
                "code": """
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(42)

# Define hyperparameters
num_epochs = 20
batch_size = 64
learning_rate = 0.001
num_classes = 10

# Data transformations with augmentation for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data transformations for validation/testing (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=test_transform
)

# Create data loaders with workers for parallel loading
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Store class names for later visualization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define CNN model with modern architecture
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# Initialize model, loss function, and optimizer
model = ConvNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total

# Training and validation function
def train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_steps = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        avg_train_acc = train_acc / train_steps
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        avg_val_acc = val_acc / val_steps
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch+1}/{num_epochs} - New best model saved! Validation Accuracy: {avg_val_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return train_losses, val_losses, train_accs, val_accs, model

# Train and validate the model
train_losses, val_losses, train_accs, val_accs, model = train_and_validate(
    model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs
)

# Plot training and validation metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Evaluate the model on test set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Function to visualize model predictions
def visualize_predictions(model, test_loader, class_names, num_images=10):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 10))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    return
                
                images_so_far += 1
                ax = plt.subplot(2, 5, images_so_far)
                ax.set_title(f'Pred: {class_names[preds[j]]}\nTrue: {class_names[labels[j]]}')
                
                # Convert image for display
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                
                ax.imshow(inp)
                ax.axis('off')
                
                if images_so_far == num_images:
                    break

# Visualize some predictions
visualize_predictions(model, test_loader, class_names)
plt.tight_layout()
plt.savefig('sample_predictions.png')
plt.show()

# Save the final model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_acc': train_accs,
    'val_acc': val_accs,
}, 'cifar10_model_checkpoint.pth')

print("Training complete! Model saved.")
""",
                "explanation": "This example demonstrates a comprehensive PyTorch implementation for image classification using convolutional neural networks. The code incorporates best practices such as data augmentation, batch normalization, dropout regularization, and proper weight initialization to improve model performance and generalization. The network architecture follows the VGG-style pattern with increasingly deeper convolutional blocks, followed by global pooling and fully connected layers. Training includes learning rate scheduling with ReduceLROnPlateau to automatically adjust the learning rate when performance plateaus. The implementation also features proper model evaluation with confusion matrices and classification reports to understand performance across different classes, along with visualization of sample predictions to provide qualitative assessment. Checkpointing is implemented to save the best model based on validation accuracy, preventing overfitting by selecting the optimal stopping point. The code is organized into modular functions for training, validation, and visualization, making it easy to adapt for different datasets or model architectures while maintaining best practices for reproducibility with fixed random seeds and proper device handling for GPU acceleration."
            },
            {
                "task": "Text Classification with scikit-learn",
                "description": "Building a text classification pipeline with preprocessing, feature extraction, and model selection using scikit-learn. This example demonstrates a complete workflow for text analysis including text cleaning, TF-IDF vectorization, hyperparameter tuning with grid search, and model evaluation.",
                "language": "Python",
                "code": """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import time
import pickle
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Text preprocessing function
def preprocess_text(text):
    \"\"\"
    Preprocess text by performing:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    \"\"\"
    # Handle non-string input
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\\w\\s]', '', text)
    text = re.sub(r'\\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return ' '.join(processed_tokens)

# Load or create sample data
def create_sample_data():
    \"\"\"
    Create a sample dataset for text classification.
    In a real scenario, you'd load your own dataset.
    \"\"\"
    categories = ['technology', 'sports', 'politics', 'entertainment']
    texts = [
        "The new smartphone has an amazing camera and long battery life.",
        "The team won the championship in a thrilling final match.",
        "The president announced new policies addressing climate change.",
        "The movie festival showcased award-winning independent films.",
        "The startup developed an AI-powered app for language translation.",
        "The athlete broke the world record during the Olympic competition.",
        "The governor signed legislation to reduce carbon emissions.",
        "The music awards ceremony featured stunning performances.",
        "New technology helps doctors diagnose diseases more accurately.",
        "The basketball tournament attracted viewers from around the world.",
        "Senators debated the tax reform bill throughout the session.",
        "The actor received critical acclaim for the challenging role.",
        "The tech company unveiled its latest smart home devices.",
        "The coach implemented a new strategy for the upcoming season.",
        "Election results showed a significant shift in voter preferences.",
        "The streaming service released an original series to critical acclaim."
    ]
    
    # Assign categories (for demonstration purposes)
    labels = [
        'technology', 'sports', 'politics', 'entertainment',
        'technology', 'sports', 'politics', 'entertainment',
        'technology', 'sports', 'politics', 'entertainment',
        'technology', 'sports', 'politics', 'entertainment'
    ]
    
    return pd.DataFrame({'text': texts, 'category': labels})

# Create dataframe
df = create_sample_data()
logger.info(f"Dataset created with {len(df)} examples and {df['category'].nunique()} categories")

# Exploratory data analysis
logger.info("Category distribution:")
category_counts = df['category'].value_counts()
logger.info(category_counts)

# Preprocess text
logger.info("Preprocessing text data...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Display examples of original and processed text
for i in range(min(3, len(df))):
    logger.info(f"Original: {df['text'][i]}")
    logger.info(f"Processed: {df['processed_text'][i]}")
    logger.info("---")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], 
    df['category'], 
    test_size=0.25, 
    random_state=42,
    stratify=df['category']  # Ensure balanced classes in train/test sets
)

logger.info(f"Training set: {len(X_train)} examples")
logger.info(f"Test set: {len(X_test)} examples")

# Create pipelines for different classifiers
pipelines = {
    'logistic_regression': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    
    'random_forest': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
}

# Parameters for grid search
parameters = {
    'logistic_regression': {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [1, 2],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'saga']
    },
    
    'random_forest': {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
}

# Perform grid search for each pipeline
best_models = {}
for name, pipeline in pipelines.items():
    logger.info(f"Tuning hyperparameters for {name}...")
    start_time = time.time()
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline, 
        parameters[name], 
        cv=5, 
        scoring='accuracy', 
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Save best model
    best_models[name] = grid_search.best_estimator_
    
    # Print results
    logger.info(f"Best parameters for {name}:")
    logger.info(grid_search.best_params_)
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
    logger.info("---")

# Evaluate the best model from each type
for name, model in best_models.items():
    logger.info(f"Evaluating {name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    logger.info(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=sorted(df['category'].unique()),
        yticklabels=sorted(df['category'].unique())
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name}.png')
    
    # If TF-IDF vectorizer is used, show top features for each class
    if hasattr(model, 'named_steps') and 'tfidf' in model.named_steps:
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['classifier']
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Show top features for each class (for models that support it)
        if hasattr(classifier, 'coef_'):
            logger.info("Top features per class:")
            
            # For each class
            for i, category in enumerate(classifier.classes_):
                # Sort features by importance
                top_features_idx = classifier.coef_[i].argsort()[-10:][::-1]
                top_features = [(feature_names[j], classifier.coef_[i][j]) for j in top_features_idx]
                
                logger.info(f"Class '{category}':")
                for feature, coef in top_features:
                    logger.info(f"  {feature}: {coef:.4f}")
    
    logger.info("---")

# Determine best overall model
best_model_name = max(best_models, key=lambda name: best_models[name].score(X_test, y_test))
best_model = best_models[best_model_name]
logger.info(f"Best overall model: {best_model_name}")

# Save the best model
with open(f'text_classifier_{best_model_name}.pkl', 'wb') as f:
    pickle.dump(best_model, f)
logger.info(f"Model saved as 'text_classifier_{best_model_name}.pkl'")

# Function to make predictions on new text
def predict_category(text, model=best_model):
    \"\"\"
    Predict category for new text
    
    Args:
        text: Input text to classify
        model: Trained model to use
    
    Returns:
        Predicted category
    \"\"\"
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Make prediction
    return model.predict([processed_text])[0]

# Function to get probability estimates (if model supports it)
def predict_proba(text, model=best_model):
    \"\"\"
    Get probability estimates for each class
    
    Args:
        text: Input text to classify
        model: Trained model to use
    
    Returns:
        Probability estimates for each class
    \"\"\"
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Make prediction with probabilities
    return model.predict_proba([processed_text])[0]

# Model deployment considerations
logger.info("Deployment Considerations:")
logger.info("1. Regularly retrain the model with new data")
logger.info("2. Monitor model performance over time for drift")
logger.info("3. Set up a preprocessing pipeline to handle new inputs consistently")
logger.info("4. Consider a feedback loop to collect misclassifications for model improvement")
""",
                "explanation": "This example demonstrates a comprehensive text classification pipeline using scikit-learn. The code showcases a complete workflow from data preparation to model deployment considerations. It features robust text preprocessing with NLTK, including tokenization, stopword removal, and lemmatization to create clean, normalized text features. The implementation uses a modular pipeline approach with TF-IDF vectorization to convert text into numerical features suitable for machine learning algorithms. The example implements model selection through GridSearchCV with hyperparameter tuning across multiple model types (Logistic Regression and Random Forest), enabling systematic comparison of different approaches. The code includes comprehensive evaluation metrics with classification reports and confusion matrices, as well as feature importance analysis to understand which terms most influence classification decisions. The implementation follows best practices with logging, proper train-test splitting with stratification to maintain class balance, and model persistence using pickle for deployment. The example concludes with practical utility functions for making predictions on new text and considerations for deploying the model in production environments, addressing common challenges like model drift and feedback collection."
            },
            {
                "task": "Time Series Forecasting with LSTM",
                "description": "Building and training an LSTM network for time series prediction, featuring data preparation, model design, training with validation, and forecasting future values. This implementation demonstrates best practices for handling sequential data and includes performance visualization.",
                "language": "Python",
                "code": """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os
import time
import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create output directory
OUTPUT_DIR = 'time_series_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to generate synthetic time series data
def generate_time_series(n_samples=1500):
    """
    Generate synthetic time series data with trend, seasonality, and noise
    
    Args:
        n_samples: Number of time steps to generate
        
    Returns:
        DataFrame with time series data
    """
    time = np.arange(0, n_samples)
    
    # Create components
    trend = 0.05 * time
    seasonality_annual = 10 * np.sin(2 * np.pi * time / 365)
    seasonality_weekly = 5 * np.sin(2 * np.pi * time / 7)
    noise = 2 * np.random.normal(0, 1, n_samples)
    
    # Combine components
    signal = trend + seasonality_annual + seasonality_weekly + noise
    
    # Create a dataframe
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),
        'value': signal
    }).set_index('timestamp')
    
    return df

# Function to load or generate data
def load_data(data_path=None, n_samples=1500):
    """
    Load time series data from file or generate synthetic data
    
    Args:
        data_path: Path to CSV file with time series data
        n_samples: Number of samples to generate if no file provided
        
    Returns:
        DataFrame with time series data
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        return df
    else:
        logger.info("Generating synthetic time series data")
        return generate_time_series(n_samples)

# Function to create sequences
def create_sequences(data, seq_length):
    """
    Create input/output sequences for time series forecasting
    
    Args:
        data: Array of time series values
        seq_length: Number of time steps to use as input
        
    Returns:
        X: Input sequences [samples, seq_length, features]
        y: Target values
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Function to build and compile LSTM model
def build_lstm_model(sequence_length, n_features=1, lstm_units=50, dropout_rate=0.2):
    """
    Build LSTM model for time series forecasting
    
    Args:
        sequence_length: Number of time steps in input sequence
        n_features: Number of features in input data
        lstm_units: Number of LSTM units in hidden layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, 
             input_shape=(sequence_length, n_features)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    return model

# Function to train and evaluate model
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train LSTM model with callbacks for early stopping and best model saving
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Training history and trained model
    """
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    train_time = time.time() - start_time
    logger.info(f"Model training completed in {train_time:.2f} seconds")
    
    return history, model

# Function for model evaluation
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate model on test data
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data (scaled)
        scaler: Scaler used to transform data
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    # Store and print metrics
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    logger.info("Model Evaluation:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")
    
    return metrics

# Function to visualize results
def plot_results(history, y_test, y_pred, scaler, dates=None):
    """
    Plot training history and test predictions
    
    Args:
        history: Training history from model.fit
        y_test, y_pred: Test actuals and predictions (scaled)
        scaler: Scaler used to transform data
        dates: Date indices for test data (optional)
    """
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
    
    # Plot test predictions vs actual values
    plt.figure(figsize=(12, 6))
    
    # Inverse transform data
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Plot with dates if available
    if dates is not None:
        plt.plot(dates, y_test_inv, label='Actual')
        plt.plot(dates, y_pred_inv, label='Predicted')
        plt.xlabel('Date')
    else:
        plt.plot(y_test_inv, label='Actual')
        plt.plot(y_pred_inv, label='Predicted')
        plt.xlabel('Time Steps')
    
    plt.title('Time Series Forecast: Actual vs Predicted')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'test_predictions.png'))
    plt.close()
    
    # Plot prediction error
    plt.figure(figsize=(12, 6))
    error = y_test_inv.flatten() - y_pred_inv.flatten()
    
    if dates is not None:
        plt.plot(dates, error)
        plt.xlabel('Date')
    else:
        plt.plot(error)
        plt.xlabel('Time Steps')
    
    plt.title('Prediction Error')
    plt.ylabel('Error')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_error.png'))
    plt.close()

# Function to forecast future values
def forecast_future(model, last_sequence, n_steps, scaler):
    """
    Generate future predictions
    
    Args:
        model: Trained LSTM model
        last_sequence: Last sequence from the data
        n_steps: Number of steps to predict
        scaler: Scaler used for normalization
        
    Returns:
        DataFrame with future predictions
    """
    # Make a copy of the last sequence
    curr_sequence = last_sequence.copy()
    future_predictions = []
    
    # Generate predictions one by one
    for _ in range(n_steps):
        # Reshape for prediction
        curr_sequence_reshaped = curr_sequence.reshape(1, curr_sequence.shape[0], 1)
        
        # Get prediction (next step)
        next_pred = model.predict(curr_sequence_reshaped)[0]
        
        # Store prediction
        future_predictions.append(next_pred[0])
        
        # Update sequence (remove first element, append prediction)
        curr_sequence = np.append(curr_sequence[1:], next_pred)
    
    # Convert to array and invert scaling
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions

# Function to run the full time series forecasting workflow
def run_time_series_workflow(data_path=None, sequence_length=60, forecast_horizon=30):
    """
    Run the full time series forecasting workflow
    
    Args:
        data_path: Path to data file (optional)
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of future steps to predict
        
    Returns:
        Dictionary with model, metrics, and predictions
    """
    # Step 1: Load or generate data
    df = load_data(data_path)
    
    # Step 2: Visualize original data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['value'])
    plt.title('Original Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'original_data.png'))
    plt.close()
    
    # Step 3: Prepare data
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['value']].values)
    
    # Create sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    # Split data into train, validation, and test sets
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    test_dates = df.index[train_size+val_size+sequence_length:]
    
    # Step 4: Build and train model
    model = build_lstm_model(
        sequence_length=sequence_length,
        n_features=1,
        lstm_units=50,
        dropout_rate=0.2
    )
    
    history, trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=32
    )
    
    # Step 5: Evaluate model
    y_pred = trained_model.predict(X_test)
    metrics = evaluate_model(trained_model, X_test, y_test, scaler)
    
    # Step 6: Visualize results
    plot_results(history, y_test, y_pred, scaler, test_dates)
    
    # Step 7: Forecast future values
    last_sequence = X_test[-1]
    future_predictions = forecast_future(
        model=trained_model,
        last_sequence=last_sequence,
        n_steps=forecast_horizon,
        scaler=scaler
    )
    
    # Create future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq='D'
    )
    
    future_df = pd.DataFrame({
        'date': future_dates,
        'prediction': future_predictions.flatten()
    })
    
    # Save future predictions
    future_df.to_csv(os.path.join(OUTPUT_DIR, 'future_predictions.csv'), index=False)
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 100 points)
    hist_dates = df.index[-100:]
    hist_values = df['value'].values[-100:]
    plt.plot(hist_dates, hist_values, label='Historical Data')
    
    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Predictions', color='red', marker='o', markersize=3)
    
    # Mark the forecast start
    plt.axvline(x=last_date, color='green', linestyle='--', label='Forecast Start')
    
    plt.title(f'Time Series Forecast: Next {forecast_horizon} Days')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'future_predictions.png'))
    plt.close()
    
    # Step 8: Save model and artifacts
    model_path = os.path.join(OUTPUT_DIR, 'final_model.h5')
    trained_model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaling parameters
    import pickle
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save workflow metadata
    metadata = {
        'sequence_length': sequence_length,
        'forecast_horizon': forecast_horizon,
        'training_size': len(X_train),
        'validation_size': len(X_val),
        'test_size': len(X_test),
        'metrics': metrics,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=4)
    
    logger.info("Time series forecasting workflow completed successfully!")
    
    return {
        'model': trained_model,
        'metrics': metrics,
        'future_predictions': future_df,
        'metadata': metadata
    }

# Run the workflow if script is executed directly
if __name__ == "__main__":
    run_time_series_workflow(sequence_length=60, forecast_horizon=30)
    """,
                "explanation": "This example demonstrates a comprehensive time series forecasting implementation using LSTM neural networks in TensorFlow. The code begins with systematic data preparation, including normalization using MinMaxScaler to ensure all values fall into the appropriate range for neural network training, and sequence creation that transforms the original time series into supervised learning format with rolling windows of input-output pairs. The LSTM model architecture follows best practices with stacked layers to capture hierarchical patterns, dropout for regularization to prevent overfitting, and proper sequence handling. Training incorporates several important techniques including early stopping to prevent overfitting, learning rate reduction when performance plateaus, and model checkpointing to save the best version. The evaluation section provides multiple complementary metrics (RMSE, MAE, R²) to thoroughly assess model performance, along with visualizations of predictions versus actuals and error analysis to identify potential patterns in prediction inaccuracies. A key feature is the recursive multi-step forecasting function that enables predicting arbitrary future time horizons by feeding predictions back into the model sequentially. The workflow concludes with structured artifact management, saving the trained model, scaling parameters, and metadata to enable seamless deployment and reproducibility."
            },
            {
                "task": "Fine-tuning BERT for Sentiment Analysis",
                "description": "Fine-tuning a pre-trained BERT model for sentiment classification with Hugging Face Transformers, including data preparation, model configuration, training with learning rate scheduling, and evaluation with detailed metrics.",
                "language": "Python",
                "code": """
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import logging
import json
import time
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bert_sentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

set_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Create directories for outputs
OUTPUT_DIR = 'bert_sentiment_output'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class SentimentDataset(Dataset):
    """Dataset for sentiment analysis with BERT"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_sentiment_data(data_file=None, random_state=42):
    """
    Load sentiment data from a file or create synthetic data
    """
    if data_file and os.path.exists(data_file):
        logger.info(f"Loading data from {data_file}")
        return pd.read_csv(data_file)
    
    # Create synthetic data for demonstration
    logger.info("Creating synthetic sentiment data")
    
    # Positive examples
    positive_texts = [
        "This movie was fantastic! I really enjoyed it.",
        "The service at the restaurant was excellent and the food was delicious.",
        "I love this product, it exceeded all my expectations.",
        "The hotel was clean, comfortable, and the staff was very friendly.",
        "This book is engaging and well-written, I couldn't put it down.",
        "The experience was wonderful from start to finish.",
        "Amazing customer service and very quick delivery.",
        "The performance was outstanding, best concert I've been to.",
        "Highly recommend this, it's definitely worth the price.",
        "The app is intuitive and has all the features I need."
    ]
    
    # Negative examples
    negative_texts = [
        "This movie was terrible, complete waste of time.",
        "The service was slow and the food was cold when it arrived.",
        "The product broke after two days, very disappointing quality.",
        "The hotel room was dirty and the staff was unhelpful.",
        "This book is boring and poorly written, couldn't finish it.",
        "The experience was frustrating and not worth the hassle.",
        "Poor customer service and the item arrived damaged.",
        "The performance was below average, I expected much better.",
        "Would not recommend, definitely not worth the price.",
        "The app keeps crashing and is missing basic features."
    ]
    
    # Generate more data by combining parts
    texts = []
    labels = []
    
    # Generate positive reviews
    for _ in range(50):
        if random.random() < 0.7:
            # Use an existing positive review
            texts.append(random.choice(positive_texts))
        else:
            # Generate a new positive review by combining parts
            parts = [
                random.choice(["Great", "Excellent", "Amazing", "Wonderful", "Fantastic"]),
                random.choice(["product", "service", "experience", "quality", "value"]),
                random.choice(["!", ".", "!!", ". Highly recommend.", ". Will buy again."])
            ]
            texts.append(" ".join(parts))
        labels.append(1)
    
    # Generate negative reviews
    for _ in range(50):
        if random.random() < 0.7:
            # Use an existing negative review
            texts.append(random.choice(negative_texts))
        else:
            # Generate a new negative review by combining parts
            parts = [
                random.choice(["Terrible", "Poor", "Disappointing", "Awful", "Bad"]),
                random.choice(["product", "service", "experience", "quality", "value"]),
                random.choice(["!", ".", "!!", ". Would not recommend.", ". Will not buy again."])
            ]
            texts.append(" ".join(parts))
        labels.append(0)
    
    # Create DataFrame and shuffle
    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Save the synthetic data
    df.to_csv(os.path.join(OUTPUT_DIR, 'synthetic_sentiment_data.csv'), index=False)
    
    return df

class BertSentimentAnalyzer:
    """
    BERT-based sentiment analyzer
    """
    def __init__(self, 
                 num_labels=2, 
                 max_length=128,
                 batch_size=16,
                 learning_rate=2e-5,
                 epsilon=1e-8,
                 num_epochs=4,
                 warmup_proportion=0.1,
                 model_name='bert-base-uncased'):
        """
        Initialize BERT sentiment analyzer
        
        Args:
            num_labels: Number of sentiment classes
            max_length: Maximum sequence length
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            epsilon: Epsilon for Adam optimizer
            num_epochs: Number of training epochs
            warmup_proportion: Proportion of training steps for learning rate warmup
            model_name: Pre-trained BERT model name
        """
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.warmup_proportion = warmup_proportion
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def prepare_data(self, df, text_col='text', label_col='label', test_size=0.2, val_size=0.1):
        """
        Prepare data for training and evaluation
        
        Args:
            df: DataFrame with text and labels
            text_col: Column name for text
            label_col: Column name for labels
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
        
        Returns:
            train_loader, val_loader, test_loader
        """
        logger.info("Preparing data for BERT fine-tuning...")
        
        # Split train and test
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42, 
            stratify=df[label_col]
        )
        
        # Split train and validation
        train_df, val_df = train_test_split(
            train_df, 
            test_size=val_size/(1-test_size), 
            random_state=42, 
            stratify=train_df[label_col]
        )
        
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        # Create datasets
        train_dataset = SentimentDataset(
            texts=train_df[text_col].values,
            labels=train_df[label_col].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        val_dataset = SentimentDataset(
            texts=val_df[text_col].values,
            labels=val_df[label_col].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        test_dataset = SentimentDataset(
            texts=test_df[text_col].values,
            labels=test_df[label_col].values,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size
        )
        
        val_loader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.batch_size
        )
        
        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=self.batch_size
        )
        
        # Store data for later use
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        return train_loader, val_loader, test_loader
    
    def build_model(self):
        """
        Build the BERT model for sequence classification
        """
        logger.info(f"Building BERT model with {self.num_labels} labels...")
        
        # Load pre-trained model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Move model to device
        self.model = self.model.to(device)
        
        return self.model
    
    def train(self, train_loader, val_loader):
        """
        Train the BERT model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training metrics
        """
        logger.info("Starting BERT training...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.epsilon
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * self.num_epochs
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.warmup_proportion),
            num_training_steps=total_steps
        )
        
        # Variables for tracking best model
        best_val_accuracy = 0.0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_accuracy = 0
            
            # Progress bar for training
            progress_bar = tqdm(train_loader, desc=f"Training - Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # Calculate accuracy
                preds = torch.argmax(logits, dim=1)
                acc = (preds == batch['labels']).float().mean()
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                train_accuracy += acc.item()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "accuracy": f"{acc.item():.4f}"})
            
            # Calculate average training metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_train_accuracy = train_accuracy / len(train_loader)
            
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(avg_train_accuracy)
            
            logger.info(f"Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Training Accuracy: {avg_train_accuracy:.4f}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_accuracy = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    # Calculate accuracy
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == batch['labels']).float().mean()
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_accuracy += acc.item()
                    
                    # Store predictions for metrics
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(batch['labels'].cpu().numpy())
            
            # Calculate average validation metrics
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(avg_val_accuracy)
            
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Accuracy: {avg_val_accuracy:.4f}")
            
            # Compute additional metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_true, val_preds, average='weighted'
            )
            
            logger.info(f"Validation Precision: {precision:.4f}")
            logger.info(f"Validation Recall: {recall:.4f}")
            logger.info(f"Validation F1: {f1:.4f}")
            
            # Save best model
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                best_model_state = self.model.state_dict().copy()
                logger.info(f"New best model with validation accuracy: {best_val_accuracy:.4f}")
        
        # Load best model if validation was performed
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        # Return training statistics
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def evaluate(self, test_loader):
        """
        Evaluate the BERT model on test data
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        
        self.model.eval()
        test_loss = 0
        test_accuracy = 0
        all_preds = []
        all_probs = []
        all_true = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # Get predictions and probabilities
                preds = torch.argmax(logits, dim=1)
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Update metrics
                test_loss += loss.item()
                test_accuracy += (preds == batch['labels']).float().mean().item()
                
                # Store predictions, probabilities, and true labels
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_true.extend(batch['labels'].cpu().numpy())
        
        # Calculate average test metrics
        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / len(test_loader)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, all_preds, average='weighted'
        )
        
        # Create confusion matrix
        cm = confusion_matrix(all_true, all_preds)
        
        # Log metrics
        logger.info(f"Test Loss: {avg_test_loss:.4f}")
        logger.info(f"Test Accuracy: {avg_test_accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1: {f1:.4f}")
        
        # Return all metrics
        metrics = {
            'loss': avg_test_loss,
            'accuracy': avg_test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'true_labels': all_true.tolist()
        }
        
        return metrics
    
    def predict(self, texts):
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to classify
        
        Returns:
            Tuple of (predicted_labels, predicted_probabilities)
        """
        self.model.eval()
        
        # Create dataset for prediction
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        token_type_ids = encodings['token_type_ids'].to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            
            # Get predictions and probabilities
            preds = torch.argmax(logits, dim=1)
            probs = torch.nn.functional.softmax(logits, dim=1)
        
        return preds.cpu().numpy(), probs.cpu().numpy()
    
    def save_model(self, output_dir):
        """
        Save model, tokenizer, and configuration
        
        Args:
            output_dir: Output directory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model_weights.pt'))
        
        # Save model configuration
        self.model.config.to_json_file(os.path.join(output_dir, 'config.json'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training history
        history = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'train_accuracies': [float(x) for x in self.train_accuracies],
            'val_accuracies': [float(x) for x in self.val_accuracies]
        }
        
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        # Save model metadata
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, path):
        """
        Load a saved model from disk
        
        Args:
            path: Path to saved model
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        
        # Try to load metadata
        metadata_path = os.path.splitext(path)[0] + '_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata loaded from {metadata_path}")
            return metadata
        
        return None
    
    def plot_training_history(self, filepath=None):
        """
        Plot training history
        
        Args:
            filepath: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save to file if filepath is provided
        if filepath:
            plt.savefig(filepath)
            plt.close()
        else:
            plt.show()

# Function to train BERT classifier for sentiment analysis
def train_bert_sentiment(data_file=None, num_epochs=4, batch_size=16, learning_rate=2e-5, max_length=128):
    """
    Train BERT model for sentiment analysis
    
    Args:
        data_file: Path to data file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        max_length: Maximum sequence length
    
    Returns:
        Trained model and evaluation metrics
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load data
    df = load_sentiment_data(data_file)
    
    # Create analyzer
    analyzer = BertSentimentAnalyzer(
        num_labels=2,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = analyzer.prepare_data(df)
    
    # Build model
    analyzer.build_model()
    
    # Train model
    logger.info("Training BERT model...")
    start_time = time.time()
    
    training_stats = analyzer.train(train_loader, val_loader)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model on test set...")
    metrics = analyzer.evaluate(test_loader)
    
    # Save model
    analyzer.save_model(MODEL_DIR)
    
    # Plot training history
    analyzer.plot_training_history(os.path.join(OUTPUT_DIR, 'training_history.png'))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.array(metrics['confusion_matrix']), 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Return analyzer and metrics
    return analyzer, metrics

# Example usage
if __name__ == "__main__":
    # Train BERT sentiment analyzer
    analyzer, metrics = train_bert_sentiment(
        num_epochs=3,
        batch_size=16
    )
    
    # Print test metrics
    logger.info("Test metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Example predictions
    example_texts = [
        "I absolutely loved this product! It exceeded all my expectations.",
        "The customer service was terrible and the product arrived damaged.",
        "It's okay, not great but not terrible either."
    ]
    
    # Make predictions
    labels, probs = analyzer.predict(example_texts)
    
    # Print predictions
    logger.info("Example predictions:")
    for i, text in enumerate(example_texts):
        sentiment = "Positive" if labels[i] == 1 else "Negative"
        confidence = probs[i, labels[i]]
        
        logger.info(f"Text: '{text}'")
        logger.info(f"Prediction: {sentiment} (Confidence: {confidence:.4f})")
        logger.info("---")        }
""",
                "explanation": "This implementation demonstrates fine-tuning a pre-trained BERT model for sentiment analysis using PyTorch and the Hugging Face Transformers library. The code follows a structured approach with a dedicated BertSentimentAnalyzer class that handles the full workflow from data preparation to model deployment. The data preparation stage includes creating a custom PyTorch Dataset that efficiently tokenizes text using BERT's subword tokenizer and handles batching with appropriate padding and truncation. The training procedure incorporates several best practices for transformer fine-tuning, including the AdamW optimizer which corrects weight decay implementation, a linear learning rate scheduler with warmup to help stabilize early training, and gradient clipping to prevent exploding gradients. The implementation also features robust evaluation with comprehensive metrics (accuracy, precision, recall, F1 score) and visualization tools for analyzing model performance through confusion matrices and training curves. A particularly valuable feature is the model checkpointing system that saves the best model based on validation performance, preventing overfitting by selecting the optimal stopping point. Finally, the code provides practical utilities for making predictions on new texts with confidence scores and model persistence for deployment. This implementation would be suitable for a wide range of sentiment analysis applications in customer feedback analysis, social media monitoring, product reviews, and other text classification tasks requiring nuanced understanding of sentiment."
            },
            {
                "task": "Hyperparameter Optimization with Optuna",
                "description": "Using Optuna to systematically optimize hyperparameters for a machine learning model, featuring efficient search space definition, pruning unpromising trials, and visualization of optimization results with detailed analysis.",
                "language": "Python",
                "code": """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
from optuna.visualization import plot_slice, plot_contour
import sklearn
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create output directory
OUTPUT_DIR = 'optuna_optimization_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna
    """
    def __init__(self, problem_type='regression', random_state=42):
        """
        Initialize hyperparameter optimizer
        
        Args:
            problem_type: 'regression' or 'classification'
            random_state: Random seed for reproducibility
        """
        self.problem_type = problem_type
        self.random_state = random_state
        
        # Set up available models based on problem type
        if problem_type == 'regression':
            self.models = {
                'random_forest': RandomForestRegressor,
                'gradient_boosting': GradientBoostingRegressor,
                'xgboost': xgb.XGBRegressor,
                'lightgbm': lgb.LGBMRegressor,
                'elastic_net': ElasticNet,
                'svr': SVR,
                'knn': KNeighborsRegressor,
                'mlp': MLPRegressor
            }
            self.scoring = 'neg_root_mean_squared_error'
        elif problem_type == 'classification':
            self.models = {
                'random_forest': RandomForestClassifier,
                'xgboost': xgb.XGBClassifier,
                'lightgbm': lgb.LGBMClassifier,
                'svc': SVC,
                'knn': KNeighborsClassifier,
                'mlp': MLPClassifier
            }
            self.scoring = 'roc_auc'
        else:
            raise ValueError("Problem type must be either 'regression' or 'classification'")
        
        # Set default attributes
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_params = None
        self.study = None
        self.feature_names = None
        
        np.random.seed(random_state)
    
    def load_data(self, X=None, y=None, dataset_name=None, test_size=0.2):
        """
        Load or prepare data for optimization
        
        Args:
            X: Feature matrix (optional)
            y: Target vector (optional)
            dataset_name: Name of built-in dataset to load (optional)
            test_size: Proportion of data to use for testing
        """
        if X is not None and y is not None:
            logger.info("Using provided data")
            self.X = X
            self.y = y
        elif dataset_name:
            logger.info(f"Loading {dataset_name} dataset")
            if dataset_name == 'diabetes':
                data = load_diabetes()
                self.problem_type = 'regression'
                self.scoring = 'neg_root_mean_squared_error'
            elif dataset_name == 'breast_cancer':
                data = load_breast_cancer()
                self.problem_type = 'classification'
                self.scoring = 'roc_auc'
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names.tolist()
        else:
            raise ValueError("Either provide X and y or specify a dataset_name")
        
        # Split data into train and test sets
        if self.problem_type == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=self.random_state
            )
        
        logger.info(f"Data loaded: X_train shape {self.X_train.shape}, y_train shape {self.y_train.shape}")
        logger.info(f"Class distribution (train): {np.bincount(self.y_train) if self.problem_type == 'classification' else 'N/A'}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def optimize(self, model_type, n_trials=100, study_name=None, direction='maximize', n_jobs=-1):
        """
        Run hyperparameter optimization
        
        Args:
            model_type: Type of model to optimize ('random_forest', 'xgboost', etc.)
            n_trials: Number of optimization trials
            study_name: Name for the study
            direction: 'maximize' or 'minimize'
            n_jobs: Number of parallel jobs (-1 for all processors)
        """
        if study_name is None:
            study_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting hyperparameter optimization for {model_type} with {n_trials} trials")
        
        # Verify model type is available
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not available for {self.problem_type} problems")
        
        # Store current model type
        self.model_type = model_type
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        # Create the objective function based on model type
        objective = self._create_objective(model_type)
        
        # Run optimization
        self.study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        
        # Get best parameters and value
        self.best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Best {self.scoring}: {best_value:.4f}")
        logger.info("Best hyperparameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        # Train final model with best parameters
        self._train_final_model()
        
        # Save optimization results
        self._save_results()
        
        # Create visualizations
        self._create_visualizations()
        
        return self.best_params, best_value
    
    def _create_objective(self, model_type):
        """
        Create the objective function for Optuna
        
        Args:
            model_type: Type of model to optimize
        
        Returns:
            Callable objective function
        """
        # Define search space and training procedure for each model type
        if model_type == 'random_forest':
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 32),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': self.random_state
                }
                return self._evaluate_params(params, model_type, trial)
        
        elif model_type == 'gradient_boosting':
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                    'random_state': self.random_state
                }
                return self._evaluate_params(params, model_type, trial)
        
        elif model_type == 'xgboost':
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                    'random_state': self.random_state
                }
                
                # Add objective for classification problems
                if self.problem_type == 'classification':
                    params['objective'] = 'binary:logistic'
                    params['eval_metric'] = 'auc'
                
                return self._evaluate_params(params, model_type, trial)
        
        elif model_type == 'lightgbm':
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'max_depth': trial.suggest_int('max_depth', -1, 15),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                    'random_state': self.random_state
                }
                
                # Add metric for classification problems
                if self.problem_type == 'classification':
                    params['objective'] = 'binary'
                    params['metric'] = 'auc'
                
                return self._evaluate_params(params, model_type, trial)
        
        elif model_type in ['elastic_net', 'svr']:
            def objective(trial):
                if model_type == 'elastic_net':
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.0001, 1.0, log=True),
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                        'max_iter': 2000,
                        'random_state': self.random_state
                    }
                else:  # SVR
                    params = {
                        'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                        'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
                        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    }
                
                # Use a pipeline with scaling for these models
                steps = [
                    ('scaler', StandardScaler()),
                    ('model', self.models[model_type](**params))
                ]
                pipeline = Pipeline(steps)
                
                return self._evaluate_pipeline(pipeline, trial)
        
        elif model_type == 'svc':
            def objective(trial):
                params = {
                    'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'probability': True,
                    'random_state': self.random_state
                }
                
                # Use a pipeline with scaling for SVC
                steps = [
                    ('scaler', StandardScaler()),
                    ('model', self.models[model_type](**params))
                ]
                pipeline = Pipeline(steps)
                
                return self._evaluate_pipeline(pipeline, trial)
        
        elif model_type == 'knn':
            def objective(trial):
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'p': trial.suggest_int('p', 1, 2),  # Manhattan or Euclidean
                    'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
                }
                
                # Use a pipeline with scaling for KNN
                steps = [
                    ('scaler', StandardScaler()),
                    ('model', self.models[model_type](**params))
                ]
                pipeline = Pipeline(steps)
                
                return self._evaluate_pipeline(pipeline, trial)
        
        elif model_type == 'mlp':
            def objective(trial):
                params = {
                    'hidden_layer_sizes': trial.suggest_categorical(
                        'hidden_layer_sizes',
                        [(50,), (100,), (50, 50), (100, 50), (100, 100)]
                    ),
                    'activation': trial.suggest_categorical(
                        'activation', ['relu', 'tanh', 'logistic']
                    ),
                    'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
                    'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                    'learning_rate_init': trial.suggest_float(
                        'learning_rate_init', 0.0001, 0.1, log=True
                    ),
                    'max_iter': 1000,
                    'random_state': self.random_state
                }
                
                # Use a pipeline with scaling for MLP
                steps = [
                    ('scaler', StandardScaler()),
                    ('model', self.models[model_type](**params))
                ]
                pipeline = Pipeline(steps)
                
                return self._evaluate_pipeline(pipeline, trial)
        
        else:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        return objective
    
    def _evaluate_params(self, params, model_type, trial):
        """
        Evaluate a set of hyperparameters using cross-validation
        
        Args:
            params: Dictionary of hyperparameters
            model_type: Type of model
            trial: Optuna trial object
        
        Returns:
            Mean cross-validation score
        """
        # Create model with parameters
        model = self.models[model_type](**params)
        
        # Set up cross-validation
        if self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        scores = cross_val_score(
            model, self.X_train, self.y_train, cv=cv, scoring=self.scoring, n_jobs=-1
        )
        
        # Calculate mean score
        mean_score = scores.mean()
        
        # Report intermediate result
        trial.report(mean_score, step=0)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return mean_score
    
    def _evaluate_pipeline(self, pipeline, trial):
        """
        Evaluate a pipeline using cross-validation
        
        Args:
            pipeline: Scikit-learn pipeline
            trial: Optuna trial object
        
        Returns:
            Mean cross-validation score
        """
        # Set up cross-validation
        if self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        scores = cross_val_score(
            pipeline, self.X_train, self.y_train, cv=cv, scoring=self.scoring, n_jobs=-1
        )
        
        # Calculate mean score
        mean_score = scores.mean()
        
        # Report intermediate result
        trial.report(mean_score, step=0)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return mean_score
    
    def _train_final_model(self):
        """
        Train the final model using the best hyperparameters
        """
        logger.info("Training final model with best hyperparameters")
        
        # Create model with best parameters
        if self.model_type in ['svr', 'svc', 'knn', 'mlp']:
            # These models need scaling
            steps = [
                ('scaler', StandardScaler()),
                ('model', self.models[self.model_type](**self.best_params))
            ]
            self.best_model = Pipeline(steps)
        else:
            self.best_model = self.models[self.model_type](**self.best_params)
        
        # Train on the entire training set
        self.best_model.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        self._evaluate_final_model()
    
    def _evaluate_final_model(self):
        """
        Evaluate the final model on the test set
        """
        # Make predictions
        if self.problem_type == 'classification':
            # For classification, get both class predictions and probabilities
            if hasattr(self.best_model, 'predict_proba'):
                y_prob = self.best_model.predict_proba(self.X_test)
                # For binary classification, get positive class probability
                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                    
                # Calculate ROC AUC
                roc_auc = roc_auc_score(self.y_test, y_prob)
                logger.info(f"Test ROC AUC: {roc_auc:.4f}")
                
                # Store metrics
                self.test_metrics = {'roc_auc': roc_auc}
            
            # Get class predictions
            y_pred = self.best_model.predict(self.X_test)
            
            # Calculate accuracy and F1
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"Test F1 Score: {f1:.4f}")
            
            # Update metrics
            self.test_metrics.update({
                'accuracy': accuracy,
                'f1_score': f1
            })
            
        else:  # Regression
            # Make predictions
            y_pred = self.best_model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            
            logger.info(f"Test RMSE: {rmse:.4f}")
            
            # Store metrics
            self.test_metrics = {'rmse': rmse}
    
    def _save_results(self):
        """
        Save optimization results and final model
        """
        # Create results directory
        results_dir = os.path.join(OUTPUT_DIR, f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save study
        study_path = os.path.join(results_dir, 'study.pkl')
        joblib.dump(self.study, study_path)
        
        # Save best model
        model_path = os.path.join(results_dir, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'problem_type': self.problem_type,
            'best_params': self.best_params,
            'best_value': float(self.study.best_value),
            'test_metrics': self.test_metrics,
            'feature_names': self.feature_names,
            'n_trials': len(self.study.trials),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Results saved to {results_dir}")
    
    def _create_visualizations(self):
        """
        Create visualizations of the optimization results
        """
        # Create results directory
        results_dir = os.path.join(OUTPUT_DIR, f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Optimization history
        plt.figure(figsize=(10, 6))
        plot_optimization_history(self.study)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'optimization_history.png'))
        
        # 2. Parameter importances
        plt.figure(figsize=(10, 6))
        plot_param_importances(self.study)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'param_importances.png'))
        
        # 3. Slice plot for top parameters
        try:
            plt.figure(figsize=(12, 10))
            plot_slice(self.study)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'param_slices.png'))
        except:
            logger.warning("Could not create slice plot")
        
        # 4. Contour plot for two most important parameters
        try:
            plt.figure(figsize=(10, 8))
            plot_contour(self.study)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'param_contour.png'))
        except:
            logger.warning("Could not create contour plot")
        
        # 5. Parallel coordinate plot
        param_names = list(self.best_params.keys())
        if len(param_names) >= 3:
            try:
                from optuna.visualization import plot_parallel_coordinate
                plt.figure(figsize=(12, 8))
                plot_parallel_coordinate(self.study, params=param_names)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'parallel_coordinate.png'))
            except:
                logger.warning("Could not create parallel coordinate plot")
        
        # 6. Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_') or (
            hasattr(self.best_model, 'named_steps') and 
            hasattr(self.best_model.named_steps['model'], 'feature_importances_')
        ):
            plt.figure(figsize=(10, 8))
            
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            else:
                importances = self.best_model.named_steps['model'].feature_importances_
            
            # Use feature names if available
            if self.feature_names is not None:
                features = self.feature_names
            else:
                features = [f'feature_{i}' for i in range(len(importances))]
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            sorted_features = [features[i] for i in indices]
            sorted_importances = importances[indices]
            
            # Plot
            plt.barh(range(len(sorted_features)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance for Best Model')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
        
        # 7. Predictions vs Actual for regression
        if self.problem_type == 'regression':
            plt.figure(figsize=(10, 8))
            y_pred = self.best_model.predict(self.X_test)
            
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'))
            
            # Residual plot
            plt.figure(figsize=(10, 8))
            residuals = self.y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='k', linestyle='--', lw=2)
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'residual_plot.png'))
        
        # 8. Confusion matrix for classification
        if self.problem_type == 'classification':
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            
            plt.figure(figsize=(10, 8))
            y_pred = self.best_model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        
        logger.info(f"Visualizations saved to {results_dir}")
    
    def predict(self, X):
        """
        Make predictions with the best model
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.best_model.predict(X)

# Example usage
if __name__ == "__main__":
    # Create optimizer for a regression problem
    optimizer = HyperparameterOptimizer(problem_type='regression')
    
    # Load data
    optimizer.load_data(dataset_name='diabetes')
    
    # Run optimization for a specific model type
    best_params, best_score = optimizer.optimize(
        model_type='xgboost',
        n_trials=50,
        study_name='xgboost_optimization'
    )
    
    # Make predictions with the best model
    predictions = optimizer.predict(optimizer.X_test)
    
    print("Optimization completed!")
    """,
                "explanation": "This implementation provides a comprehensive hyperparameter optimization framework using Optuna, a powerful library designed for efficient parameter search. The HyperparameterOptimizer class serves as a unified interface for optimizing various machine learning models across both regression and classification problems. The code implements intelligent search spaces for different model types, configuring appropriate parameter ranges based on algorithm characteristics, and uses Optuna's Tree-structured Parzen Estimator (TPE) sampler which balances exploration and exploitation more effectively than random or grid search. A key feature is the integration of pruning mechanisms via MedianPruner, which automatically terminates unpromising trials to focus computational resources on more promising parameter combinations. The evaluation methodology employs proper cross-validation with stratification for classification tasks to ensure robust performance estimates across folds. For algorithms sensitive to feature scaling like SVMs and neural networks, the implementation automatically creates pipelines that include standardization before model fitting. The framework also handles both regression and classification metrics appropriately, selecting relevant scoring functions based on the problem type. After optimization completes, a final model is trained using the best parameters and thoroughly evaluated on the held-out test set. Extensive visualization capabilities offer insights into the optimization process, parameter importance, and model performance through various plots that help understand which parameters most influence the model's effectiveness. All results, models, and metadata are systematically saved for reproducibility and future reference, making this a production-ready solution for automated hyperparameter tuning."
            },
            {
                "task": "MLOps Pipeline with Model Monitoring",
                "description": "Building a complete MLOps pipeline with model training, validation, deployment, and drift monitoring to track model performance and data distribution changes over time.",
                "language": "Python",
                "code": """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import ks_2samp, wasserstein_distance, chi2_contingency
import joblib
import json
import os
import yaml
import logging
import time
import datetime
import uuid
import pickle
import shutil
import subprocess
import warnings
import flask
from flask import Flask, request, jsonify
import threading
import schedule
from typing import Dict, List, Union, Optional, Tuple, Any
warnings.filterwarnings('ignore')

# Create directory structure
BASE_DIR = 'mlops_pipeline'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
MONITORING_DIR = os.path.join(BASE_DIR, 'monitoring')

# Create directories
for directory in [BASE_DIR, MODELS_DIR, DATA_DIR, LOGS_DIR, CONFIG_DIR, MONITORING_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'mlops_pipeline.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'model': {
        'type': 'random_forest',
        'problem_type': 'classification',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    },
    'data': {
        'train_data_path': os.path.join(DATA_DIR, 'train.csv'),
        'validation_data_path': os.path.join(DATA_DIR, 'validation.csv'),
        'test_data_path': os.path.join(DATA_DIR, 'test.csv'),
        'target_column': 'target',
        'features_to_exclude': ['id', 'timestamp']
    },
    'preprocessing': {
        'numeric_features': [],  # Auto-detect if empty
        'categorical_features': [],  # Auto-detect if empty
        'handle_missing_values': True,
        'scale_numeric_features': True
    },
    'training': {
        'validation_strategy': 'hold_out',  # 'hold_out' or 'cross_validation'
        'cv_folds': 5,
        'test_size': 0.2,
        'random_state': 42
    },
    'monitoring': {
        'enable_monitoring': True,
        'data_drift_threshold': 0.1,
        'performance_threshold': 0.1,
        'monitoring_frequency': 'daily',  # 'hourly', 'daily', 'weekly'
        'retraining_frequency': 'weekly',  # 'daily', 'weekly', 'monthly'
        'metrics_to_monitor': [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        ]
    },
    'serving': {
        'api_port': 5000,
        'batch_size': 1000,
        'enable_explanations': True
    }
}

# Save default configuration
with open(os.path.join(CONFIG_DIR, 'default_config.yaml'), 'w') as f:
    yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)

class DataProcessor:
    """
    Handles data loading, validation, preprocessing, and feature engineering
    """
    def __init__(self, config):
        """
        Initialize the data processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config['preprocessing']
        self.data_config = config['data']
        self.target_column = self.data_config['target_column']
        self.features_to_exclude = self.data_config.get('features_to_exclude', [])
        
        # Initialize preprocessing components
        self.preprocessor = None
        self.feature_names = None
    
    def load_data(self, data_path=None):
        """
        Load data from specified path or use default training data
        
        Args:
            data_path: Path to data CSV file
        
        Returns:
            X: Features DataFrame
            y: Target Series (if available)
        """
        if data_path is None:
            data_path = self.data_config['train_data_path']
        
        logger.info(f"Loading data from {data_path}")
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Check if target column exists
            if self.target_column in df.columns:
                # Split features and target
                y = df[self.target_column]
                X = df.drop(columns=[self.target_column] + self.features_to_exclude)
                
                return X, y
            else:
                # For inference data without target
                X = df.drop(columns=self.features_to_exclude, errors='ignore')
                return X, None
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_preprocessing_pipeline(self, X):
        """
        Create preprocessing pipeline based on data and configuration
        
        Args:
            X: Features DataFrame to analyze for preprocessing
        
        Returns:
            Preprocessor pipeline
        """
        # Auto-detect feature types if not specified
        numeric_features = self.preprocessing_config.get('numeric_features', [])
        categorical_features = self.preprocessing_config.get('categorical_features', [])
        
        if not numeric_features:
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not categorical_features:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")
        
        # Create transformers for each feature type
        transformers = []
        
        # Numeric features pipeline
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')) if self.preprocessing_config.get('handle_missing_values', True) else ('imputer', 'passthrough'),
                ('scaler', StandardScaler()) if self.preprocessing_config.get('scale_numeric_features', True) else ('scaler', 'passthrough')
            ])
            
            transformers.append(('numeric', numeric_transformer, numeric_features))
        
        # Categorical features pipeline
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')) if self.preprocessing_config.get('handle_missing_values', True) else ('imputer', 'passthrough'),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            transformers.append(('categorical', categorical_transformer, categorical_features))
        
        # Create column transformer
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # Store feature names for later use
        self.feature_names = numeric_features + categorical_features
        
        return preprocessor
    
    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess data using the preprocessing pipeline
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            fit: Whether to fit the preprocessor or just transform
        
        Returns:
            X_processed: Processed features
            y: Target (unchanged)
        """
        # Create preprocessor if not already created
        if self.preprocessor is None or fit:
            self.preprocessor = self.prepare_preprocessing_pipeline(X)
        
        # Apply preprocessing
        if fit:
            logger.info("Fitting and transforming data")
            X_processed = self.preprocessor.fit_transform(X)
        else:
            logger.info("Transforming data using existing preprocessor")
            X_processed = self.preprocessor.transform(X)
        
        return X_processed, y
    
    def save_preprocessor(self, path):
        """
        Save the preprocessing pipeline to disk
        
        Args:
            path: Path to save the preprocessor
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized. Run preprocess_data first.")
        
        # Save preprocessor
        joblib.dump(self.preprocessor, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path):
        """
        Load preprocessor from disk
        
        Args:
            path: Path to saved preprocessor
        """
        self.preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")

class ModelTrainer:
    """
    Handles model training, validation, and evaluation
    """
    def __init__(self, config):
        """
        Initialize the model trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.problem_type = self.model_config['problem_type']
        
        # Initialize model
        self.model = None
        self.model_type = self.model_config.get('type', 'random_forest')
        self.model_params = self.model_config.get('params', {})
    
    def create_model(self):
        """
        Create a model based on configuration
        
        Returns:
            Model instance
        """
        model_type = self.model_type.lower()
        
        if self.problem_type == 'classification':
            if model_type == 'random_forest':
                return RandomForestClassifier(**self.model_params)
            elif model_type == 'xgboost':
                return xgb.XGBClassifier(**self.model_params)
            elif model_type == 'lightgbm':
                return lgb.LGBMClassifier(**self.model_params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        elif self.problem_type == 'regression':
            if model_type == 'random_forest':
                return RandomForestRegressor(**self.model_params)
            elif model_type == 'xgboost':
                return xgb.XGBRegressor(**self.model_params)
            elif model_type == 'lightgbm':
                return lgb.LGBMRegressor(**self.model_params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on the provided data
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        
        Returns:
            Trained model, training metrics
        """
        logger.info(f"Training {self.model_type} model")
        
        # Create model if not already created
        if self.model is None:
            self.model = self.create_model()
        
        # Train the model
        start_time = time.time()
        
        if self.training_config['validation_strategy'] == 'cross_validation':
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=self.training_config['cv_folds'], shuffle=True, random_state=self.training_config['random_state']) if self.problem_type == 'classification' else None
            
            # Use appropriate scoring metric
            if self.problem_type == 'classification':
                scoring = 'roc_auc'
            else:
                scoring = 'neg_root_mean_squared_error'
            
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
            
            # Train final model on all data
            self.model.fit(X_train, y_train)
            
            metrics = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean_score': float(cv_scores.mean()),
                'cv_std_score': float(cv_scores.std())
            }
            
            logger.info(f"Cross-validation {scoring}: {metrics['cv_mean_score']:.4f} ± {metrics['cv_std_score']:.4f}")
        
        else:
            # Hold-out validation
            self.model.fit(X_train, y_train)
            
            if X_val is not None and y_val is not None:
                # Evaluate on validation set
                metrics = self.evaluate_model(X_val, y_val)
                logger.info(f"Validation metrics: {metrics}")
            else:
                # No validation data provided
                metrics = {}
        
        # Record training time
        training_time = time.time() - start_time
        metrics['training_time'] = training_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return self.model, metrics
    
    def evaluate_model(self, X, y):
        """
        Evaluate the model on the provided data
        
        Args:
            X: Features
            y: Target
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Make predictions
        if self.problem_type == 'classification':
            # Get predicted probabilities and classes
            y_pred_proba = self.model.predict_proba(X)
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, average='weighted')),
                'recall': float(recall_score(y, y_pred, average='weighted')),
                'f1': float(f1_score(y, y_pred, average='weighted')),
            }
            
            # Add ROC AUC if binary classification
            if len(np.unique(y)) == 2:
                metrics['roc_auc'] = float(roc_auc_score(y, y_pred_proba[:, 1]))
            
            # Add confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
            
        else:  # Regression
            # Get predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'mae': float(mean_absolute_error(y, y_pred)),
                'r2': float(r2_score(y, y_pred))
            }
        
        return metrics
    
    def save_model(self, path, metadata=None):
        """
        Save the trained model to disk
        
        Args:
            path: Path to save the model
            metadata: Additional metadata to save with the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.splitext(path)[0] + '_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a saved model from disk
        
        Args:
            path: Path to saved model
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        
        # Try to load metadata
        metadata_path = os.path.splitext(path)[0] + '_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata loaded from {metadata_path}")
            return metadata
        
        return None
    
    def get_feature_importances(self, feature_names=None):
        """
        Get feature importances from the model
        
        Args:
            feature_names: Names of features (optional)
        
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Check if model supports feature importances
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importances")
            return None
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Use feature names if provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

class ModelDeployer:
    """
    Handles model deployment, versioning, and serving
    """
    def __init__(self, config):
        """
        Initialize the model deployer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.serving_config = config['serving']
        self.api_port = self.serving_config.get('api_port', 5000)
        self.batch_size = self.serving_config.get('batch_size', 1000)
        
        # Initialize model and preprocessor
        self.model = None
        self.preprocessor = None
        self.model_metadata = None
        self.model_version = None
        self.deployment_id = None
        
        # Initialize API server
        self.api_server = None
        self.api_thread = None
    
    def deploy_model(self, model_path, preprocessor_path, model_version=None):
        """
        Deploy a trained model for serving
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
            model_version: Version identifier for the model
        """
        # Generate deployment ID
        self.deployment_id = str(uuid.uuid4())
        
        # Set model version
        if model_version is None:
            self.model_version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.model_version = model_version
        
        # Load model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load model metadata if available
        metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        
        # Create deployment directory
        deployment_dir = os.path.join(MODELS_DIR, f'deployment_{self.model_version}')
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Copy model and preprocessor to deployment directory
        shutil.copy(model_path, os.path.join(deployment_dir, 'model.joblib'))
        shutil.copy(preprocessor_path, os.path.join(deployment_dir, 'preprocessor.joblib'))
        
        # Save deployment metadata
        deployment_metadata = {
            'deployment_id': self.deployment_id,
            'model_version': self.model_version,
            'deployment_time': datetime.datetime.now().isoformat(),
            'model_metadata': self.model_metadata
        }
        
        with open(os.path.join(deployment_dir, 'deployment_metadata.json'), 'w') as f:
            json.dump(deployment_metadata, f, indent=4)
        
        logger.info(f"Model deployed with version {self.model_version} and ID {self.deployment_id}")
        
        # Start API server
        self.start_api_server()
        
        return self.deployment_id
    
    def start_api_server(self):
        """
        Start a Flask API server for model serving
        """
        # Create Flask app
        app = Flask(__name__)
        
        # Define prediction endpoint
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                # Get data from request
                data = request.json
                
                # Convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
                
                # Make prediction
                predictions = self.predict(df)
                
                return jsonify({
                    'predictions': predictions,
                    'model_version': self.model_version,
                    'deployment_id': self.deployment_id
                })
            
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'model_version': self.model_version,
                    'deployment_id': self.deployment_id
                }), 400
        
        # Define health check endpoint
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'ok',
                'model_version': self.model_version,
                'deployment_id': self.deployment_id
            })
        
        # Define metadata endpoint
        @app.route('/metadata', methods=['GET'])
        def metadata():
            return jsonify({
                'model_version': self.model_version,
                'deployment_id': self.deployment_id,
                'model_metadata': self.model_metadata
            })
        
        # Store app reference
        self.api_server = app
        
        # Start server in a separate thread
        def run_server():
            app.run(host='0.0.0.0', port=self.api_port)
        
        self.api_thread = threading.Thread(target=run_server)
        self.api_thread.daemon = True
        self.api_thread.start()
        
        logger.info(f"API server started on port {self.api_port}")
    
    def predict(self, data):
        """
        Make predictions using the deployed model
        
        Args:
            data: Input data for prediction
        
        Returns:
            Predictions
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not deployed yet. Call deploy_model first.")
        
        # Preprocess data
        X_processed = self.preprocessor.transform(data)
        
        # Make prediction
        if hasattr(self.model, 'predict_proba'):
            # Classification with probabilities
            y_proba = self.model.predict_proba(X_processed)
            y_pred = self.model.predict(X_processed)
            
            # Create result
            result = []
            for i in range(len(y_pred)):
                pred = {
                    'prediction': int(y_pred[i]) if isinstance(y_pred[i], (np.int64, np.int32)) else y_pred[i],
                    'probabilities': {str(j): float(p) for j, p in enumerate(y_proba[i])}
                }
                result.append(pred)
        
        else:
            # Regression or classification without probabilities
            y_pred = self.model.predict(X_processed)
            
            # Create result
            result = [{'prediction': float(p) if isinstance(p, (np.float64, np.float32)) else p} for p in y_pred]
        
        return result
    
    def batch_predict(self, data):
        """
        Make predictions in batches
        
        Args:
            data: Input data for prediction
        
        Returns:
            Predictions
        """
        # Process data in batches
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data.iloc[i:i+self.batch_size]
            batch_results = self.predict(batch)
            results.extend(batch_results)
        
        return results
    
    def load_deployment(self, deployment_dir):
        """
        Load a saved deployment
        
        Args:
            deployment_dir: Path to deployment directory
        """
        # Load model and preprocessor
        model_path = os.path.join(deployment_dir, 'model.joblib')
        preprocessor_path = os.path.join(deployment_dir, 'preprocessor.joblib')
        metadata_path = os.path.join(deployment_dir, 'deployment_metadata.json')
        
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            deployment_metadata = json.load(f)
        
        self.model_version = deployment_metadata['model_version']
        self.deployment_id = deployment_metadata['deployment_id']
        self.model_metadata = deployment_metadata.get('model_metadata', {})
        
        logger.info(f"Deployment loaded from {deployment_dir}")
        
        # Start API server
        self.start_api_server()
        
        return deployment_metadata

class ModelMonitor:
    """
    Monitors model performance and data drift
    """
    def __init__(self, config):
        """
        Initialize the model monitor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.monitoring_config = config['monitoring']
        self.problem_type = config['model']['problem_type']
        
        # Monitoring parameters
        self.data_drift_threshold = self.monitoring_config.get('data_drift_threshold', 0.1)
        self.performance_threshold = self.monitoring_config.get('performance_threshold', 0.1)
        self.metrics_to_monitor = self.monitoring_config.get('metrics_to_monitor', [])
        
        # Reference data and metrics
        self.reference_data = None
        self.reference_metrics = None
        self.reference_distributions = None
        
        # Monitoring history
        self.monitoring_history = []
    
    def set_reference_data(self, data, metrics=None):
        """
        Set reference data for drift detection
        
        Args:
            data: Reference dataset
            metrics: Reference model performance metrics
        """
        self.reference_data = data
        self.reference_metrics = metrics
        
        # Calculate reference feature distributions
        self.reference_distributions = self._calculate_distributions(data)
        
        # Save reference data and distributions
        reference_dir = os.path.join(MONITORING_DIR, 'reference')
        os.makedirs(reference_dir, exist_ok=True)
        
        # Save data sample (if not too large)
        if len(data) <= 10000:
            data.to_csv(os.path.join(reference_dir, 'reference_data.csv'), index=False)
        else:
            # Save a sample
            data.sample(10000, random_state=42).to_csv(
                os.path.join(reference_dir, 'reference_data_sample.csv'), index=False
            )
        
        # Save distributions
        with open(os.path.join(reference_dir, 'reference_distributions.json'), 'w') as f:
            json.dump(self._serialize_distributions(self.reference_distributions), f, indent=4)
        
        # Save metrics if provided
        if metrics is not None:
            with open(os.path.join(reference_dir, 'reference_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
        
        logger.info("Reference data and distributions saved")
    
    def detect_drift(self, current_data, current_metrics=None):
        """
        Detect data and performance drift
        
        Args:
            current_data: Current dataset to compare against reference
            current_metrics: Current model performance metrics
        
        Returns:
            Drift detection results
        """
        if self.reference_data is None or self.reference_distributions is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        # Calculate current distributions
        current_distributions = self._calculate_distributions(current_data)
        
        # Calculate data drift metrics
        data_drift_metrics = self._calculate_drift_metrics(
            self.reference_distributions, current_distributions
        )
        
        # Calculate performance drift if metrics provided
        performance_drift = {}
        if current_metrics is not None and self.reference_metrics is not None:
            for metric in self.metrics_to_monitor:
                if metric in current_metrics and metric in self.reference_metrics:
                    ref_value = self.reference_metrics[metric]
                    current_value = current_metrics[metric]
                    
                    # Calculate relative change
                    if ref_value != 0:
                        relative_change = abs(current_value - ref_value) / abs(ref_value)
                    else:
                        relative_change = abs(current_value - ref_value)
                    
                    performance_drift[metric] = {
                        'reference_value': ref_value,
                        'current_value': current_value,
                        'absolute_change': current_value - ref_value,
                        'relative_change': relative_change,
                        'drift_detected': relative_change > self.performance_threshold
                    }
        
        # Determine overall drift status
        data_drift_detected = any(m['drift_detected'] for m in data_drift_metrics.values())
        performance_drift_detected = any(m['drift_detected'] for m in performance_drift.values()) if performance_drift else False
        
        # Create drift detection result
        drift_result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_drift': {
                'drift_detected': data_drift_detected,
                'metrics': data_drift_metrics
            },
            'performance_drift': {
                'drift_detected': performance_drift_detected,
                'metrics': performance_drift
            },
            'overall_drift_detected': data_drift_detected or performance_drift_detected
        }
        
        # Save drift result
        self._save_drift_result(drift_result, current_distributions)
        
        # Add to monitoring history
        self.monitoring_history.append(drift_result)
        
        # Log drift detection
        if drift_result['overall_drift_detected']:
            logger.warning("Drift detected! Consider retraining the model.")
        else:
            logger.info("No significant drift detected.")
        
        return drift_result
    
    def _calculate_distributions(self, data):
        """
        Calculate distributions for numeric and categorical features
        
        Args:
            data: Dataset
        
        Returns:
            Feature distributions
        """
        distributions = {}
        
        # Process numeric features
        for col in data.select_dtypes(include=['int64', 'float64']).columns:
            # Skip columns with all missing values
            if data[col].isna().all():
                continue
            
            # Get non-missing values
            values = data[col].dropna().values
            
            # Skip empty columns
            if len(values) == 0:
                continue
            
            # Calculate distribution statistics
            distributions[col] = {
                'type': 'numeric',
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'q1': float(np.percentile(values, 25)),
                'q3': float(np.percentile(values, 75)),
                'histogram': np.histogram(values, bins=10)
            }
        
        # Process categorical features
        for col in data.select_dtypes(include=['object', 'category']).columns:
            # Skip columns with all missing values
            if data[col].isna().all():
                continue
            
            # Calculate value counts
            value_counts = data[col].value_counts(normalize=True, dropna=True)
            
            # Skip empty columns
            if len(value_counts) == 0:
                continue
            
            distributions[col] = {
                'type': 'categorical',
                'unique_values': len(value_counts),
                'frequencies': value_counts.to_dict()
            }
        
        return distributions
    
    def _serialize_distributions(self, distributions):
        """
        Serialize distributions for JSON storage
        
        Args:
            distributions: Feature distributions
        
        Returns:
            Serialized distributions
        """
        serialized = {}
        
        for col, dist in distributions.items():
            serialized[col] = dist.copy()
            
            # Convert numpy arrays in histograms to lists
            if dist['type'] == 'numeric' and 'histogram' in dist:
                serialized[col]['histogram'] = {
                    'counts': dist['histogram'][0].tolist(),
                    'bins': dist['histogram'][1].tolist()
                }
        
        return serialized
    
    def _deserialize_distributions(self, serialized):
        """
        Deserialize distributions from JSON storage
        
        Args:
            serialized: Serialized distributions
        
        Returns:
            Feature distributions
        """
        distributions = {}
        
        for col, dist in serialized.items():
            distributions[col] = dist.copy()
            
            # Convert histogram lists back to numpy arrays
            if dist['type'] == 'numeric' and 'histogram' in dist:
                distributions[col]['histogram'] = (
                    np.array(dist['histogram']['counts']),
                    np.array(dist['histogram']['bins'])
                )
        
        return distributions
    
    def _calculate_drift_metrics(self, reference_distributions, current_distributions):
        """
        Calculate drift metrics between reference and current distributions
        
        Args:
            reference_distributions: Reference feature distributions
            current_distributions: Current feature distributions
        
        Returns:
            Drift metrics
        """
        drift_metrics = {}
        
        # Check each feature present in both distributions
        for col in set(reference_distributions.keys()) & set(current_distributions.keys()):
            ref_dist = reference_distributions[col]
            curr_dist = current_distributions[col]
            
            # Ensure same type
            if ref_dist['type'] != curr_dist['type']:
                logger.warning(f"Feature {col} has changed type from {ref_dist['type']} to {curr_dist['type']}")
                continue
            
            # Calculate drift metrics based on feature type
            if ref_dist['type'] == 'numeric':
                # Use statistical tests for numeric features
                
                # Extract values from histograms
                ref_counts, ref_bins = ref_dist['histogram']
                curr_counts, curr_bins = curr_dist['histogram']
                
                # Calculate KS statistic (normalized)
                ref_cdf = np.cumsum(ref_counts) / np.sum(ref_counts)
                curr_cdf = np.cumsum(curr_counts) / np.sum(curr_counts)
                ks_stat = np.max(np.abs(ref_cdf - curr_cdf))
                
                # Calculate Wasserstein distance (normalized)
                w_dist = wasserstein_distance(ref_cdf, curr_cdf)
                
                # Calculate distribution changes
                mean_change = abs(curr_dist['mean'] - ref_dist['mean']) / (abs(ref_dist['std']) if ref_dist['std'] != 0 else 1)
                std_change = abs(curr_dist['std'] - ref_dist['std']) / (abs(ref_dist['std']) if ref_dist['std'] != 0 else 1)
                
                # Combine metrics
                drift_score = max(ks_stat, w_dist, mean_change, std_change)
                
                drift_metrics[col] = {
                    'type': 'numeric',
                    'ks_statistic': float(ks_stat),
                    'wasserstein_distance': float(w_dist),
                    'mean_change': float(mean_change),
                    'std_change': float(std_change),
                    'drift_score': float(drift_score),
                    'drift_detected': drift_score > self.data_drift_threshold
                }
            
            else:  # Categorical
                # Compare frequency distributions
                ref_freqs = ref_dist['frequencies']
                curr_freqs = curr_dist['frequencies']
                
                # Get all categories
                all_categories = set(ref_freqs.keys()) | set(curr_freqs.keys())
                
                # Calculate JS divergence (modified for missing categories)
                js_divergence = 0
                
                for category in all_categories:
                    p = ref_freqs.get(category, 0)
                    q = curr_freqs.get(category, 0)
                    
                    # Calculate contribution to JS divergence
                    if p > 0 and q > 0:
                        m = 0.5 * (p + q)
                        js_divergence += 0.5 * (p * np.log(p / m) + q * np.log(q / m))
                    elif p > 0:
                        js_divergence += 0.5 * p
                    elif q > 0:
                        js_divergence += 0.5 * q
                
                # Calculate chi-squared statistic
                chi2_stat = 0
                
                if len(all_categories) > 1:
                    # Create frequency arrays
                    ref_counts = np.array([ref_freqs.get(cat, 0) for cat in all_categories])
                    curr_counts = np.array([curr_freqs.get(cat, 0) for cat in all_categories])
                    
                    # Ensure non-zero counts for chi-squared test
                    if np.all(ref_counts > 0) and np.all(curr_counts > 0):
                        try:
                            chi2_stat, _ = chi2_contingency(
                                np.vstack([ref_counts, curr_counts])
                            )[:2]
                            # Normalize chi-squared
                            chi2_stat = chi2_stat / (len(all_categories) - 1) if len(all_categories) > 1 else 0
                        except:
                            chi2_stat = 0
                
                # Calculate new categories ratio
                new_categories = set(curr_freqs.keys()) - set(ref_freqs.keys())
                new_cat_ratio = len(new_categories) / max(len(all_categories), 1)
                
                # Combine metrics
                drift_score = max(js_divergence, chi2_stat, new_cat_ratio)
                
                drift_metrics[col] = {
                    'type': 'categorical',
                    'js_divergence': float(js_divergence),
                    'chi2_statistic': float(chi2_stat),
                    'new_categories_ratio': float(new_cat_ratio),
                    'new_categories': list(new_categories),
                    'drift_score': float(drift_score),
                    'drift_detected': drift_score > self.data_drift_threshold
                }
        
        return drift_metrics
    
    def _save_drift_result(self, drift_result, current_distributions):
        """
        Save drift detection result
        
        Args:
            drift_result: Drift detection result
            current_distributions: Current feature distributions
        """
        # Create timestamp-based directory
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(MONITORING_DIR, f'drift_{timestamp}')
        os.makedirs(result_dir, exist_ok=True)
        
        # Save drift result
        with open(os.path.join(result_dir, 'drift_result.json'), 'w') as f:
            json.dump(drift_result, f, indent=4)
        
        # Save current distributions
        with open(os.path.join(result_dir, 'current_distributions.json'), 'w') as f:
            json.dump(self._serialize_distributions(current_distributions), f, indent=4)
        
        # Generate drift report
        drift_report = self._generate_drift_report(drift_result)
        with open(os.path.join(result_dir, 'drift_report.txt'), 'w') as f:
            f.write(drift_report)
    
    def _generate_drift_report(self, drift_result):
        """
        Generate human-readable drift report
        
        Args:
            drift_result: Drift detection result
        
        Returns:
            Drift report text
        """
        report = []
        report.append(f"DRIFT DETECTION REPORT - {drift_result['timestamp']}")
        report.append("=" * 80)
        
        # Overall drift status
        report.append(f"OVERALL DRIFT DETECTED: {drift_result['overall_drift_detected']}")
        report.append("")
        
        # Data drift section
        report.append("DATA DRIFT ANALYSIS")
        report.append("-" * 50)
        report.append(f"Data Drift Detected: {drift_result['data_drift']['drift_detected']}")
        report.append("")
        
        # Add details for features with drift
        drifted_features = [
            col for col, metrics in drift_result['data_drift']['metrics'].items()
            if metrics['drift_detected']
        ]
        
        if drifted_features:
            report.append(f"Drifted Features ({len(drifted_features)}):")
            for col in drifted_features:
                metrics = drift_result['data_drift']['metrics'][col]
                report.append(f"  - {col} (type: {metrics['type']}, drift score: {metrics['drift_score']:.4f})")
                
                if metrics['type'] == 'numeric':
                    report.append(f"    * KS statistic: {metrics['ks_statistic']:.4f}")
                    report.append(f"    * Wasserstein distance: {metrics['wasserstein_distance']:.4f}")
                    report.append(f"    * Mean change: {metrics['mean_change']:.4f}")
                    report.append(f"    * Std change: {metrics['std_change']:.4f}")
                else:  # Categorical
                    report.append(f"    * JS divergence: {metrics['js_divergence']:.4f}")
                    report.append(f"    * Chi2 statistic: {metrics['chi2_statistic']:.4f}")
                    report.append(f"    * New categories ratio: {metrics['new_categories_ratio']:.4f}")
                    if metrics['new_categories']:
                        report.append(f"    * New categories: {', '.join(metrics['new_categories'])}")
        else:
            report.append("No features with significant data drift.")
        
        report.append("")
        
        # Performance drift section
        if drift_result['performance_drift']['metrics']:
            report.append("PERFORMANCE DRIFT ANALYSIS")
            report.append("-" * 50)
            report.append(f"Performance Drift Detected: {drift_result['performance_drift']['drift_detected']}")
            report.append("")
            
            # Add details for metrics with drift
            drifted_metrics = [
                metric for metric, data in drift_result['performance_drift']['metrics'].items()
                if data['drift_detected']
            ]
            
            if drifted_metrics:
                report.append(f"Drifted Metrics ({len(drifted_metrics)}):")
                for metric in drifted_metrics:
                    data = drift_result['performance_drift']['metrics'][metric]
                    report.append(f"  - {metric}:")
                    report.append(f"    * Reference value: {data['reference_value']:.4f}")
                    report.append(f"    * Current value: {data['current_value']:.4f}")
                    report.append(f"    * Absolute change: {data['absolute_change']:.4f}")
                    report.append(f"    * Relative change: {data['relative_change']:.4f}")
            else:
                report.append("No metrics with significant performance drift.")
        
        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 50)
        
        if drift_result['overall_drift_detected']:
            report.append("Based on the drift analysis, the following actions are recommended:")
            
            if drift_result['data_drift']['drift_detected']:
                report.append("1. Investigate the drifted features to understand the root cause of the drift.")
                report.append("2. Consider collecting new training data that better represents the current distribution.")
                report.append("3. Retrain the model with the updated dataset.")
            
            if drift_result.get('performance_drift', {}).get('drift_detected', False):
                report.append("4. Review model performance metrics and consider model tuning or architecture changes.")
        else:
            report.append("No significant drift detected. The model appears to be performing well on the current data.")
        
        return "\n".join(report)
    
    def schedule_monitoring(self, data_loader_func, evaluator_func=None, frequency='daily'):
        """
        Schedule regular monitoring
        
        Args:
            data_loader_func: Function to load current data
            evaluator_func: Function to evaluate model performance
            frequency: Monitoring frequency ('hourly', 'daily', 'weekly')
        """
        def monitoring_job():
            logger.info(f"Running scheduled monitoring ({frequency})")
            
            try:
                # Load current data
                current_data = data_loader_func()
                
                # Evaluate performance if evaluation function provided
                current_metrics = evaluator_func() if evaluator_func else None
                
                # Detect drift
                drift_result = self.detect_drift(current_data, current_metrics)
                
                # Log result
                if drift_result['overall_drift_detected']:
                    logger.warning("Drift detected in scheduled monitoring!")
                else:
                    logger.info("No drift detected in scheduled monitoring.")
            
            except Exception as e:
                logger.error(f"Error in scheduled monitoring: {str(e)}")
        
        # Schedule job based on frequency
        if frequency == 'hourly':
            schedule.every().hour.do(monitoring_job)
        elif frequency == 'daily':
            schedule.every().day.at("00:00").do(monitoring_job)
        elif frequency == 'weekly':
            schedule.every().monday.at("00:00").do(monitoring_job)
        else:
            raise ValueError(f"Unsupported monitoring frequency: {frequency}")
        
        # Start the scheduler in a background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info(f"Monitoring scheduled with {frequency} frequency")

class MLOpsWorkflow:
    """
    Orchestrates the entire MLOps workflow
    """
    def __init__(self, config_path=None):
        """
        Initialize the MLOps workflow
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            # Use default configuration
            self.config = DEFAULT_CONFIG
            logger.info("Using default configuration")
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_deployer = ModelDeployer(self.config)
        self.model_monitor = ModelMonitor(self.config)
        
        # Set up execution environment
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up execution environment"""
        # Create necessary directories
        for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR, CONFIG_DIR, MONITORING_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Save current configuration
        config_path = os.path.join(CONFIG_DIR, 'current_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def train_and_evaluate(self, train_data_path=None, validation_data_path=None):
        """
        Train and evaluate a model
        
        Args:
            train_data_path: Path to training data
            validation_data_path: Path to validation data
        
        Returns:
            Model, performance metrics
        """
        # Override paths if provided
        if train_data_path:
            self.config['data']['train_data_path'] = train_data_path
        
        if validation_data_path:
            self.config['data']['validation_data_path'] = validation_data_path
        
        # Load and preprocess training data
        logger.info("Loading training data")
        X_train, y_train = self.data_processor.load_data(self.config['data']['train_data_path'])
        X_train_processed, y_train = self.data_processor.preprocess_data(X_train, y_train)
        
        # Load and preprocess validation data if provided
        X_val = y_val = X_val_processed = None
        if validation_data_path or 'validation_data_path' in self.config['data']:
            logger.info("Loading validation data")
            val_path = validation_data_path or self.config['data']['validation_data_path']
            X_val, y_val = self.data_processor.load_data(val_path)
            X_val_processed, y_val = self.data_processor.preprocess_data(X_val, y_val, fit=False)
        
        # Train model
        logger.info("Training model")
        model, metrics = self.model_trainer.train_model(X_train_processed, y_train, X_val_processed, y_val)
        
        # Generate timestamp for model version
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save preprocessor
        preprocessor_path = os.path.join(MODELS_DIR, f'preprocessor_{timestamp}.joblib')
        self.data_processor.save_preprocessor(preprocessor_path)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f'model_{timestamp}.joblib')
        
        # Add feature names to metadata if available
        metadata = {
            'timestamp': timestamp,
            'config': self.config,
            'metrics': metrics,
            'feature_names': self.data_processor.feature_names
        }
        
        self.model_trainer.save_model(model_path, metadata)
        
        # Set reference data for monitoring
        self.model_monitor.set_reference_data(X_train, metrics)
        
        return model, metrics, {
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'timestamp': timestamp
        }
    
    def deploy_model(self, model_path=None, preprocessor_path=None):
        """
        Deploy a trained model
        
        Args:
            model_path: Path to model file
            preprocessor_path: Path to preprocessor file
        
        Returns:
            Deployment ID
        """
        # Use latest model if paths not provided
        if model_path is None or preprocessor_path is None:
            # Find latest model and preprocessor
            model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('model_') and f.endswith('.joblib')]
            preprocessor_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('preprocessor_') and f.endswith('.joblib')]
            
            if not model_files or not preprocessor_files:
                raise ValueError("No trained models available. Train a model first.")
            
            # Sort by timestamp
            model_files.sort(reverse=True)
            preprocessor_files.sort(reverse=True)
            
            model_path = os.path.join(MODELS_DIR, model_files[0])
            preprocessor_path = os.path.join(MODELS_DIR, preprocessor_files[0])
        
        # Deploy model
        deployment_id = self.model_deployer.deploy_model(model_path, preprocessor_path)
        
        return deployment_id
    
    def monitor_performance(self, data_path=None):
        """
        Monitor model performance and data drift
        
        Args:
            data_path: Path to current data
        
        Returns:
            Drift detection results
        """
        # Load current data
        if data_path is None:
            # Try to use validation data
            if 'validation_data_path' in self.config['data']:
                data_path = self.config['data']['validation_data_path']
            else:
                raise ValueError("No data path provided for monitoring.")
        
        # Load data
        X_current, y_current = self.data_processor.load_data(data_path)
        
        # Process data
        X_current_processed, y_current = self.data_processor.preprocess_data(X_current, y_current, fit=False)
        
        # Evaluate performance if target is available
        current_metrics = None
        if y_current is not None and self.model_deployer.model is not None:
            current_metrics = self.model_trainer.evaluate_model(X_current_processed, y_current)
        
        # Detect drift
        drift_result = self.model_monitor.detect_drift(X_current, current_metrics)
        
        return drift_result
    
    def run_full_workflow(self, train_data_path=None, validation_data_path=None):
        """
        Run the full MLOps workflow: train, deploy, and monitor
        
        Args:
            train_data_path: Path to training data
            validation_data_path: Path to validation data
        
        Returns:
            Workflow results
        """
        # Step 1: Train and evaluate model
        logger.info("STEP 1: Training and evaluating model")
        model, metrics, artifacts = self.train_and_evaluate(train_data_path, validation_data_path)
        
        # Step 2: Deploy model
        logger.info("STEP 2: Deploying model")
        deployment_id = self.deploy_model(artifacts['model_path'], artifacts['preprocessor_path'])
        
        # Step 3: Monitor performance
        logger.info("STEP 3: Initial performance monitoring")
        drift_result = self.monitor_performance(validation_data_path)
        
        # Step 4: Schedule monitoring if enabled
        if self.config['monitoring']['enable_monitoring']:
            logger.info("STEP 4: Setting up scheduled monitoring")
            
            # Define data loader function
            def load_monitoring_data():
                X, y = self.data_processor.load_data(validation_data_path)
                return X
            
            # Define evaluation function
            def evaluate_model():
                X, y = self.data_processor.load_data(validation_data_path)
                X_processed, _ = self.data_processor.preprocess_data(X, y, fit=False)
                return self.model_trainer.evaluate_model(X_processed, y)
            
            # Schedule monitoring
            frequency = self.config['monitoring']['monitoring_frequency']
            self.model_monitor.schedule_monitoring(load_monitoring_data, evaluate_model, frequency)
        
        # Return workflow results
        return {
            'training': {
                'metrics': metrics,
                'artifacts': artifacts
            },
            'deployment': {
                'deployment_id': deployment_id,
                'api_port': self.model_deployer.api_port
            },
            'monitoring': {
                'initial_drift': drift_result,
                'scheduled_monitoring': self.config['monitoring']['enable_monitoring']
            }
        }
    
    def shutdown(self):
        """Clean up resources"""
        logger.info("Shutting down MLOps workflow")
        # Clean up resources if needed

# Example usage
if __name__ == "__main__":
    # Initialize workflow with default configuration
    workflow = MLOpsWorkflow()
    
    # Run synthetic data generation for demonstration
    from sklearn.datasets import make_classification, make_regression
    
    def generate_synthetic_data():
        """Generate synthetic data for demo"""
        problem_type = workflow.config['model']['problem_type']
        
        if problem_type == 'classification':
            # Generate classification data
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=5,
                n_redundant=2,
                n_classes=2,
                random_state=42
            )
        else:
            # Generate regression data
            X, y = make_regression(
                n_samples=1000,
                n_features=10,
                n_informative=5,
                noise=0.1,
                random_state=42
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        # Split into train, validation, and test
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Save data
        train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(DATA_DIR, 'validation.csv'), index=False)
        test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
        
        logger.info(f"Synthetic {problem_type} data generated and saved")
    
    # Generate data if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, 'train.csv')):
        generate_synthetic_data()
    
    # Run full workflow
    results = workflow.run_full_workflow()
    
    # Print results
    print("\nMLOps Workflow Completed!")
    print(f"Model trained with metrics: {results['training']['metrics']}")
    print(f"Model deployed with ID: {results['deployment']['deployment_id']}")
    print(f"API server running on port: {results['deployment']['api_port']}")
    
    if results['monitoring']['initial_drift']['overall_drift_detected']:
        print("WARNING: Drift detected in initial monitoring!")
    else:
        print("No drift detected in initial monitoring.")
    
    print("\nTest the API with:")
    print(f"curl -X POST http://localhost:{results['deployment']['api_port']}/predict -H 'Content-Type: application/json' -d '{{\"feature_0\": 0.1, \"feature_1\": 0.2}}'")
    
    # Keep the script running to maintain the API server
    try:
        print("\nPress Ctrl+C to exit")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        workflow.shutdown()
        print("MLOps workflow shut down")""",
                "explanation": "This comprehensive MLOps pipeline implementation demonstrates a production-grade workflow for machine learning operations, covering the entire ML lifecycle from training to deployment and monitoring. The code establishes a well-structured architecture with clear separation of concerns through specialized components: DataProcessor handles data preparation and feature engineering with automatic detection of feature types and appropriate preprocessing; ModelTrainer manages model creation, training, and evaluation with support for different problem types and algorithms; ModelDeployer provides model serving capabilities with a Flask API server for real-time predictions; and ModelMonitor implements sophisticated drift detection for both data distributions and model performance. A particularly valuable aspect is the robust drift detection implementation, which employs statistical tests like Kolmogorov-Smirnov and Wasserstein distance for numeric features, and divergence measures for categorical features, enabling early detection of data distribution shifts that could impact model performance. The system features comprehensive logging, artifact management with versioning, scheduled monitoring, and graceful handling of model updates. The pipeline's design follows MLOps best practices with clear configuration management, reproducibility through metadata tracking, and automation of routine tasks. This implementation would be suitable for organizations looking to establish reliable, maintainable machine learning systems in production environments, with particular attention to monitoring and maintaining model performance over time."
            },
            {
Reinforcement Learning with Stable Baselines
This implementation demonstrates training and evaluating reinforcement learning agents using the Stable Baselines library, featuring environment setup, custom callbacks, and policy evaluation.
pythonimport os
import time
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.env_util import make_vec_env
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reinforcement_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
LOGS_DIR = "rl_logs"
MODELS_DIR = "rl_models"
RESULTS_DIR = "rl_results"

for directory in [LOGS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class CustomTradingEnvironment(gym.Env):
    """
    A custom trading environment that simulates stock trading.
    
    This is a simplified example for demonstration purposes.
    In a real-world scenario, you would use actual market data.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=10000):
        super(CustomTradingEnvironment, self).__init__()
        
        # Initialize environment variables
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.current_price = 0
        self.last_trade_price = 0
        self.total_trades = 0
        self.max_steps = len(df) - 1
        
        # Define action and observation space
        # Actions: 0 (Sell), 1 (Hold), 2 (Buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [balance, shares_held, current_price, 5 price indicators]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
    
    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.df.iloc[self.current_step]['close']
        self.last_trade_price = 0
        self.total_trades = 0
        
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: 0 (Sell), 1 (Hold), 2 (Buy)
        
        Returns:
            observation, reward, done, info
        """
        self.current_step += 1
        self.current_price = self.df.iloc[self.current_step]['close']
        
        # Execute action
        if action == 0:  # Sell
            if self.shares_held > 0:
                # Sell all shares
                self.balance += self.shares_held * self.current_price
                self.last_trade_price = self.current_price
                self.shares_held = 0
                self.total_trades += 1
        
        elif action == 2:  # Buy
            # Calculate maximum shares that can be bought
            max_shares = self.balance // self.current_price
            if max_shares > 0:
                # Buy shares
                shares_to_buy = max_shares
                self.balance -= shares_to_buy * self.current_price
                self.shares_held += shares_to_buy
                self.last_trade_price = self.current_price
                self.total_trades += 1
        
        # Calculate portfolio value
        portfolio_value = self.balance + (self.shares_held * self.current_price)
        
        # Calculate reward
        reward = (portfolio_value / self.initial_balance) - 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'portfolio_value': portfolio_value,
            'total_trades': self.total_trades
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get the current observation of the environment."""
        # Get the current row of data
        current_data = self.df.iloc[self.current_step]
        
        # Get price indicators
        indicators = [
            current_data['sma_10'],     # 10-day Simple Moving Average
            current_data['sma_30'],     # 30-day Simple Moving Average
            current_data['rsi'],        # Relative Strength Index
            current_data['macd'],       # Moving Average Convergence Divergence
            current_data['volatility']  # Volatility (standard deviation of returns)
        ]
        
        # Combine all observations
        observation = np.array([
            self.balance,
            self.shares_held,
            self.current_price,
            *indicators
        ], dtype=np.float32)
        
        return observation
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            portfolio_value = self.balance + (self.shares_held * self.current_price)
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Current price: ${self.current_price:.2f}")
            print(f"Portfolio value: ${portfolio_value:.2f}")
            print(f"Total trades: {self.total_trades}")
            print("-" * 50)

# Function to generate synthetic stock data
def generate_synthetic_stock_data(n_steps=1000, starting_price=100.0, volatility=0.05, seed=42):
    """
    Generate synthetic stock price data with indicators.
    
    Args:
        n_steps: Number of time steps
        starting_price: Initial stock price
        volatility: Price volatility
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with price and indicators
    """
    np.random.seed(seed)
    
    # Generate price data
    returns = np.random.normal(0, volatility, n_steps)
    price_data = starting_price * (1 + np.cumsum(returns))
    
    # Generate OHLC data
    dates = pd.date_range(start='2023-01-01', periods=n_steps, freq='D')
    high = price_data * (1 + np.random.uniform(0, 0.03, n_steps))
    low = price_data * (1 - np.random.uniform(0, 0.03, n_steps))
    open_prices = price_data * (1 + np.random.normal(0, 0.01, n_steps))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': price_data,
        'volume': np.random.randint(1000, 100000, n_steps)
    })
    
    # Calculate technical indicators
    # Simple Moving Averages
    df['sma_10'] = df['close'].rolling(window=10).mean().fillna(method='bfill')
    df['sma_30'] = df['close'].rolling(window=30).mean().fillna(method='bfill')
    
    # Relative Strength Index (RSI) - simplified calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Fill NA with neutral value
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    
    # Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std().fillna(method='bfill')
    
    return df

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging to Tensorboard.
    
    This callback captures training metrics and logs them to TensorBoard
    for visualization and monitoring of the training process.
    """
    def __init__(self, verbose=0):
        """
        Initialize the TensorboardCallback.
        
        Args:
            verbose: Verbosity level (0 for quiet, 1 for info)
        """
        super(TensorboardCallback, self).__init__(verbose)
        self.training_env = None
    
    def _on_training_start(self):
        """
        Setup TensorBoard logging when training starts.
        """
        self.training_env = self.model.get_env()
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter
        self.tb_formatter = None
        for fmt in output_formats:
            if hasattr(fmt, "writer") and hasattr(fmt.writer, "add_summary"):
                self.tb_formatter = fmt
                break
    
    def _on_step(self):
        """
        Log metrics to TensorBoard on each step.
        
        This method is called on each training step and logs portfolio metrics
        to TensorBoard every 100 steps.
        
        Returns:
            True to continue training
        """
        # Log additional info every 100 steps
        if self.n_calls % 100 == 0 and self.tb_formatter is not None:
            info = self.training_env.get_attr("info")[0]
            self.tb_formatter.writer.add_scalar("portfolio_value", info['portfolio_value'], self.n_calls)
            self.tb_formatter.writer.add_scalar("total_trades", info['total_trades'], self.n_calls)
        return True

def train_and_evaluate_agents(env_creator, models_to_train=['ppo', 'a2c', 'dqn'], total_timesteps=100000):
    """
    Train and evaluate multiple reinforcement learning agents.
    
    Args:
        env_creator: Function that creates the environment
        models_to_train: List of models to train ('ppo', 'a2c', 'dqn', 'sac')
        total_timesteps: Number of timesteps to train each model
    
    Returns:
        Dictionary of trained models and evaluation results
    """
    results = {}
    
    # Create vectorized environment for training
    vec_env = make_vec_env(env_creator, n_envs=1, monitor_dir=os.path.join(LOGS_DIR, "train"))
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Create separate environment for evaluation
    eval_env = make_vec_env(env_creator, n_envs=1, monitor_dir=os.path.join(LOGS_DIR, "eval"))
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name.upper()} agent...")
        
        # Create model directory
        model_dir = os.path.join(MODELS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Set up callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=os.path.join(LOGS_DIR, model_name),
            eval_freq=5000,
            n_eval_episodes=10,
            verbose=1
        )
        
        tensorboard_callback = TensorboardCallback()
        
        # Initialize model
        if model_name == 'ppo':
            model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"))
        elif model_name == 'a2c':
            model = A2C('MlpPolicy', vec_env, verbose=1, tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"))
        elif model_name == 'dqn':
            model = DQN('MlpPolicy', vec_env, verbose=1, tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"))
        elif model_name == 'sac':
            model = SAC('MlpPolicy', vec_env, verbose=1, tensorboard_log=os.path.join(LOGS_DIR, "tensorboard"))
        else:
            logger.error(f"Unknown model type: {model_name}")
            continue
        
        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, tensorboard_callback]
        )
        training_time = time.time() - start_time
        
        # Save the final model
        model_path = os.path.join(model_dir, f"final_model")
        model.save(model_path)
        
        # Save normalized environment
        env_path = os.path.join(model_dir, "vec_normalize.pkl")
        vec_env.save(env_path)
        
        # Evaluate the trained model
        logger.info(f"Evaluating {model_name.upper()} agent...")
        
        # Load best model for evaluation
        best_model_path = os.path.join(model_dir, "best_model")
        if os.path.exists(best_model_path + ".zip"):
            if model_name == 'ppo':
                best_model = PPO.load(best_model_path)
            elif model_name == 'a2c':
                best_model = A2C.load(best_model_path)
            elif model_name == 'dqn':
                best_model = DQN.load(best_model_path)
            elif model_name == 'sac':
                best_model = SAC.load(best_model_path)
            
            # Evaluate model
            mean_reward, std_reward = evaluate_policy(
                best_model, 
                eval_env, 
                n_eval_episodes=20,
                deterministic=True
            )
        else:
            # Use final model if best model not available
            mean_reward, std_reward = evaluate_policy(
                model, 
                eval_env, 
                n_eval_episodes=20,
                deterministic=True
            )
        
        # Store results
        results[model_name] = {
            'model': model,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'training_time': training_time
        }
        
        logger.info(f"{model_name.upper()} evaluation results:")
        logger.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Training time: {training_time:.2f} seconds")
    
    return results

def backtest_trading_strategy(env_creator, model, initial_balance=10000, episodes=10):
    """
    Backtest a trading strategy using a trained model.
    
    Args:
        env_creator: Function that creates the environment
        model: Trained reinforcement learning model
        initial_balance: Initial balance for trading
        episodes: Number of episodes to run
    
    Returns:
        DataFrame with backtest results
    """
    # Create environment for backtesting
    env = env_creator()
    env.reset()
    
    results = []
    
    for episode in range(episodes):
        logger.info(f"Running backtest episode {episode+1}/{episodes}")
        
        # Reset environment
        observation = env.reset()
        done = False
        
        # Episode data
        episode_data = []
        
        while not done:
            # Get action from model
            action, _ = model.predict(observation, deterministic=True)
            
            # Take step in environment
            observation, reward, done, info = env.step(action)
            
            # Store step data
            step_data = {
                'episode': episode,
                'step': info['step'],
                'balance': info['balance'],
                'shares_held': info['shares_held'],
                'current_price': info['current_price'],
                'portfolio_value': info['portfolio_value'],
                'action': action,
                'reward': reward
            }
            
            episode_data.append(step_data)
        
        # Add episode data to results
        results.extend(episode_data)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate additional metrics
    # - Returns: Daily, Cumulative, Annualized
    # - Risk metrics: Sharpe Ratio, Maximum Drawdown
    
    # Calculate daily returns
    results_df['daily_return'] = results_df.groupby('episode')['portfolio_value'].pct_change()
    
    # Calculate cumulative returns per episode
    for episode in range(episodes):
        episode_data = results_df[results_df['episode'] == episode]
        initial_value = episode_data.iloc[0]['portfolio_value']
        results_df.loc[results_df['episode'] == episode, 'cumulative_return'] = \
            results_df.loc[results_df['episode'] == episode, 'portfolio_value'] / initial_value - 1
    
    # Calculate Sharpe ratio per episode (assuming 252 trading days per year and risk-free rate of 0)
    results_df['sharpe_ratio'] = np.sqrt(252) * results_df.groupby('episode')['daily_return'].transform(
        lambda x: x.mean() / x.std() if x.std() != 0 else 0
    )
    
    # Calculate maximum drawdown per episode
    for episode in range(episodes):
        episode_data = results_df[results_df['episode'] == episode]
        cummax = episode_data['portfolio_value'].cummax()
        drawdown = (episode_data['portfolio_value'] - cummax) / cummax
        results_df.loc[results_df['episode'] == episode, 'drawdown'] = drawdown
    
    results_df['max_drawdown'] = results_df.groupby('episode')['drawdown'].transform('min')
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "backtest_results.csv")
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"Backtest results saved to {results_path}")
    
    return results_df

def visualize_backtest_results(results_df, model_name):
    """
    Visualize backtest results.
    
    Args:
        results_df: DataFrame with backtest results
        model_name: Name of the model used
    """
    # Create results directory
    viz_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-whitegrid')
    sns.set_palette("viridis")
    
    # Create multiple plots
    
    # 1. Portfolio value over time
    plt.figure(figsize=(12, 6))
    for episode in results_df['episode'].unique():
        episode_data = results_df[results_df['episode'] == episode]
        plt.plot(episode_data['step'], episode_data['portfolio_value'], 
                 label=f'Episode {episode+1}', alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'Portfolio Value Over Time - {model_name.upper()} Model')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{model_name}_portfolio_value.png"))
    
    # 2. Cumulative returns over time
    plt.figure(figsize=(12, 6))
    for episode in results_df['episode'].unique():
        episode_data = results_df[results_df['episode'] == episode]
        plt.plot(episode_data['step'], episode_data['cumulative_return'], 
                 label=f'Episode {episode+1}', alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Return (%)')
    plt.title(f'Cumulative Returns - {model_name.upper()} Model')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{model_name}_cumulative_returns.png"))
    
    # 3. Trading actions
    plt.figure(figsize=(12, 6))
    
    # Get data for a single episode (first episode)
    episode_data = results_df[results_df['episode'] == 0]
    
    # Plot price
    ax1 = plt.gca()
    ax1.plot(episode_data['step'], episode_data['current_price'], 'b-', label='Stock Price')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Stock Price ($)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot actions on a secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot buy and sell actions
    buys = episode_data[episode_data['action'] == 2]
    sells = episode_data[episode_data['action'] == 0]
    
    ax2.scatter(buys['step'], buys['current_price'], color='g', marker='^', s=100, label='Buy')
    ax2.scatter(sells['step'], sells['current_price'], color='r', marker='v', s=100, label='Sell')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Trading Actions - {model_name.upper()} Model (Episode 1)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{model_name}_trading_actions.png"))
    
    # 4. Performance metrics comparison
    plt.figure(figsize=(12, 6))
    
    # Calculate average metrics per episode
    episode_metrics = results_df.groupby('episode').agg({
        'portfolio_value': lambda x: x.iloc[-1],  # Final portfolio value
        'cumulative_return': lambda x: x.iloc[-1],  # Final cumulative return
        'sharpe_ratio': 'mean',                  # Average Sharpe ratio
        'max_drawdown': 'min'                    # Min max_drawdown (worst drawdown)
    }).reset_index()
    
    # Plot metrics
    metrics = ['portfolio_value', 'cumulative_return', 'sharpe_ratio', 'max_drawdown']
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5))
    
    for i, metric in enumerate(metrics):
        sns.barplot(x='episode', y=metric, data=episode_metrics, ax=axes[i])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_xlabel('Episode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{model_name}_performance_metrics.png"))
    
    logger.info(f"Visualization results saved to {viz_dir}")

def run_rl_trading_experiment():
    """Run a complete reinforcement learning trading experiment."""
    # 1. Generate synthetic data
    logger.info("Generating synthetic stock data...")
    stock_data = generate_synthetic_stock_data(n_steps=1000, starting_price=100.0, volatility=0.05)
    
    # Save data
    data_path = os.path.join(DATA_DIR, "synthetic_stock_data.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    stock_data.to_csv(data_path, index=False)
    logger.info(f"Synthetic stock data saved to {data_path}")
    
    # 2. Create environment creator function
    def create_trading_env():
        return CustomTradingEnvironment(stock_data)
    
    # 3. Train and evaluate models
    logger.info("Training and evaluating models...")
    results = train_and_evaluate_agents(
        env_creator=create_trading_env,
        models_to_train=['ppo', 'a2c', 'dqn'],
        total_timesteps=50000  # Reduced for demonstration
    )
    
    # 4. Select best model
    logger.info("Selecting best model...")
    best_model_name = max(results, key=lambda k: results[k]['mean_reward'])
    best_model = results[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name.upper()} with mean reward: {results[best_model_name]['mean_reward']:.2f}")
    
    # 5. Backtest the best model
    logger.info(f"Backtesting {best_model_name.upper()} model...")
    backtest_results = backtest_trading_strategy(
        env_creator=create_trading_env,
        model=best_model,
        episodes=5
    )
    
    # 6. Visualize backtest results
    logger.info("Visualizing backtest results...")
    visualize_backtest_results(backtest_results, best_model_name)
    
    # 7. Return results
    return {
        'training_results': results,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'backtest_results': backtest_results
    }

# Example usage
if __name__ == "__main__":
    # Run the complete experiment
    experiment_results = run_rl_trading_experiment()
    
    # Print summary
    print("\nReinforcement Learning Trading Experiment Completed!")
    print(f"Best model: {experiment_results['best_model_name'].upper()}")
    
    # Access individual results
    best_model = experiment_results['best_model']
    backtest_results = experiment_results['backtest_results']
    
    # Calculate overall performance
    final_portfolio_values = backtest_results.groupby('episode')['portfolio_value'].last()
    avg_final_value = final_portfolio_values.mean()
    avg_return = avg_final_value / 10000 - 1  # Assuming initial balance of 10000
    
    print(f"Average final portfolio value: ${avg_final_value:.2f}")
    print(f"Average return: {avg_return:.2%}")


Generative AI with Diffusion Models
This implementation demonstrates building and training a diffusion model for image generation, featuring denoising network architecture, diffusion process, and sampling techniques.
pythonimport os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import make_grid, save_image

from PIL import Image
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diffusion_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
DATA_DIR = "diffusion_data"
MODELS_DIR = "diffusion_models"
RESULTS_DIR = "diffusion_results"
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")

for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SAMPLES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Define network architecture
class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings for timestep encoding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """
    Basic convolutional block with residual connection.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x, time_emb):
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[..., None, None]  # [B, C] -> [B, C, 1, 1]
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Add time embedding
        h = h + time_emb
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        
        # Add residual connection
        return h + self.residual(x)

class Attention(nn.Module):
    """
    Simple attention mechanism.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        # Compute attention
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, c, h, w)
        
        return self.proj(out)

class DownBlock(nn.Module):
    """
    Downsampling block for U-Net.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, time_emb_dim)
        self.block2 = Block(out_channels, out_channels, time_emb_dim)
        self.attention = Attention(out_channels) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
    
    def forward(self, x, time_emb):
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        x = self.attention(x)
        return self.downsample(x), x

class UpBlock(nn.Module):
    """
    Upsampling block for U-Net.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.block1 = Block(out_channels * 2, out_channels, time_emb_dim)  # *2 for skip connection
        self.block2 = Block(out_channels, out_channels, time_emb_dim)
        self.attention = Attention(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x, skip_x, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        x = self.attention(x)
        return x

class UNet(nn.Module):
    """
    U-Net architecture for denoising model in diffusion process.
    """
    def __init__(self, in_channels=1, out_channels=1, channel_multipliers=(1, 2, 4), time_emb_dim=256):
        super().__init__()
        self.time_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        base_channels = 64
        channels = [base_channels]
        for mult in channel_multipliers:
            channels.append(base_channels * mult)
        
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Downsampling blocks
        self.downs = nn.ModuleList([])
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            use_attention = i >= len(channels) - 3  # Use attention in the last 2 blocks
            self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim, use_attention))
            in_ch = out_ch
        
        # Middle block
        self.mid_block1 = Block(channels[-1], channels[-1], time_emb_dim)
        self.mid_attn = Attention(channels[-1])
        self.mid_block2 = Block(channels[-1], channels[-1], time_emb_dim)
        
        # Upsampling blocks
        self.ups = nn.ModuleList([])
        in_ch = channels[-1]
        for i, out_ch in enumerate(reversed(channels[:-1])):
            use_attention = i <= 1  # Use attention in the first 2 blocks
            self.ups.append(UpBlock(in_ch, out_ch, time_emb_dim, use_attention))
            in_ch = out_ch
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1)
        )
    
    def forward(self, x, timestep):
        # Embed timestep
        t = self.time_mlp(timestep)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsampling
        skip_connections = []
        for down in self.downs:
            x, skip_x = down(x, t)
            skip_connections.append(skip_x)
        
        # Middle blocks
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # Upsampling with skip connections
        for up in self.ups:
            skip_x = skip_connections.pop()
            x = up(x, skip_x, t)
        
        # Final convolution
        return self.final_conv(x)

class DiffusionModel:
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    """
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000, loss_type='l2'):
        """
        Initialize diffusion model.
        
        Args:
            model: Noise prediction model (UNet)
            beta_start: Starting beta value for diffusion process
            beta_end: Ending beta value for diffusion process
            timesteps: Number of diffusion steps
            loss_type: Type of loss function ('l1', 'l2')
        """
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        
        # Pre-calculate diffusion process parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - the forward diffusion process.
        
        Args:
            x_0: Initial clean data
            t: Timestep
            noise: Noise to add (if None, random noise is generated)
        
        Returns:
            Noisy sample x_t
        """
        noise = torch.randn_like(x_0) if noise is None else noise
        
        # Extract coefficients at timestep t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        
        # Compute noisy sample x_t
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, t, noise=None):
        """
        Calculate loss for denoising model training.
        
        Args:
            x_0: Initial clean data
            t: Timestep
            noise: Noise to predict (if None, random noise is generated)
        
        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]
        noise = torch.randn_like(x_0) if noise is None else noise
        
        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Calculate loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        else:
            loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def sample_timesteps(self, batch_size):
        """
        Sample random timesteps for a batch.
        
        Args:
            batch_size: Number of timesteps to sample
        
        Returns:
            Tensor of timesteps
        """
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
    
    @torch.no_grad()
    def p_sample(self, x_t, t, t_index):
        """
        Sample from p(x_{t-1} | x_t) - the reverse diffusion process (one step).
        
        Args:
            x_t: Current noisy sample
            t: Current timestep
            t_index: Index of current timestep
        
        Returns:
            Sample from previous timestep x_{t-1}
        """
        betas_t = self.betas.gather(-1, t).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.gather(-1, t).reshape(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Compute mean of posterior distribution
        posterior_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Only add noise if t > 0, otherwise just return the mean
        if t_index > 0:
            variance_t = self.posterior_variance.gather(-1, t).reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return posterior_mean + torch.sqrt(variance_t) * noise
        else:
            return posterior_mean
    
    @torch.no_grad()
    def p_sample_loop(self, shape, num_samples=1):
        """
        Sample from the model multiple times (full reverse diffusion process).
        
        Args:
            shape: Shape of samples to generate
            num_samples: Number of samples to generate
        
        Returns:
            Generated samples
        """
        device = next(self.model.parameters()).device
        b = shape[0] * num_samples
        
        # Start from pure noise
        img = torch.randn(b, *shape[1:], device=device)
        imgs = []
        
        # Iteratively denoise
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i)
            imgs.append(img.cpu())
        
        return torch.stack(imgs)
    
    def train(self, dataloader, optimizer, num_epochs, save_interval=5, sample_interval=10):
        """
        Train the diffusion model.
        
        Args:
            dataloader: DataLoader with training data
            optimizer: Optimizer for model parameters
            num_epochs: Number of training epochs
            save_interval: Interval for saving model checkpoints
            sample_interval: Interval for generating samples during training
        
        Returns:
            Training losses
        """
        self.model.train()
        losses = []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Get batch
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)
                
                # Sample random timesteps
                t = self.sample_timesteps(x.shape[0])
                
                # Calculate loss
                optimizer.zero_grad()
                loss = self.p_losses(x, t)
                loss.backward()
                optimizer.step()
                
                # Track progress
                epoch_losses.append(loss.item())
                progress_bar.set_postfix({"loss": loss.item()})
            
            # Log epoch results
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            # Save model
            if (epoch + 1) % save_interval == 0 or epoch + 1 == num_epochs:
                model_path = os.path.join(MODELS_DIR, f"diffusion_model_epoch_{epoch+1}.pt")
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Generate samples
            if (epoch + 1) % sample_interval == 0 or epoch + 1 == num_epochs:
                self.model.eval()
                sample_shape = (4, 1, 28, 28)  # Generate 4 images
                samples = self.p_sample_loop(sample_shape)[-1]  # Get the final denoised samples
                
                # Save samples
                sample_grid = make_grid(samples, nrow=2, normalize=True)
                sample_path = os.path.join(SAMPLES_DIR, f"samples_epoch_{epoch+1}.png")
                save_image(sample_grid, sample_path)
                logger.info(f"Samples saved to {sample_path}")
                
                self.model.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return losses
    
    @torch.no_grad()
    def generate_samples(self, num_samples=16, image_size=28, channels=1):
        """
        Generate samples from the trained model.
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of images
            channels: Number of channels in images
        
        Returns:
            Generated samples
        """
        self.model.eval()
        
        # Generate samples
        shape = (num_samples, channels, image_size, image_size)
        samples = self.p_sample_loop(shape)[-1]  # Get the final denoised samples
        
        # Save samples
        sample_grid = make_grid(samples, nrow=int(np.sqrt(num_samples)), normalize=True)
        sample_path = os.path.join(RESULTS_DIR, f"generated_samples.png")
        save_image(sample_grid, sample_path)
        logger.info(f"Generated samples saved to {sample_path}")
        
        return samples

def load_mnist_data(batch_size=128, image_size=28):
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for DataLoader
        image_size: Size to resize images to
    
    Returns:
        Training DataLoader
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load MNIST dataset
    train_dataset = MNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=True
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"MNIST dataset loaded with {len(train_dataset)} samples")
    
    return train_loader

def load_fashion_mnist_data(batch_size=128, image_size=28):
    """
    Load Fashion-MNIST dataset and prepare it for training.
    
    Downloads the Fashion-MNIST dataset if not available locally,
    applies normalization to the range [-1, 1], and creates a DataLoader
    with the specified batch size for efficient training.
    
    Args:
        batch_size: Batch size for DataLoader
        image_size: Size to resize images to
    
    Returns:
        DataLoader: Training data loader with prepared Fashion-MNIST data
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Load Fashion-MNIST dataset
    train_dataset = FashionMNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=True
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Fashion-MNIST dataset loaded with {len(train_dataset)} samples")
    
    return train_loader

def visualize_diffusion_process(diffusion_model, num_images=10, steps=10):
    """
    Visualize the diffusion process from noise to generated images.
    
    Creates a grid of images showing the progressive denoising process
    from random noise to coherent images. Also generates a GIF animation
    of the first image in the batch.
    
    Args:
        diffusion_model: Trained diffusion model
        num_images: Number of images to generate
        steps: Number of intermediate steps to show
    
    Returns:
        None - Saves visualization files to results directory
    """
    # Set model to evaluation mode
    diffusion_model.model.eval()
    
    # Start from pure noise
    shape = (num_images, 1, 28, 28)
    img = torch.randn(shape, device=device)
    
    # Define timesteps to visualize
    ts = list(reversed(range(0, diffusion_model.timesteps, diffusion_model.timesteps // steps)))
    if ts[-1] != 0:
        ts.append(0)
    
    # Generate samples at specified timesteps
    fig, axes = plt.subplots(num_images, len(ts), figsize=(len(ts) * 2, num_images * 2))
    
    # If only one image, ensure axes is 2D
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    images = []
    
    with torch.no_grad():
        for i, t in enumerate(ts):
            # Fill t for batch
            t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
            
            # If t > 0, sample from p(x_{t-1} | x_t)
            if t > 0:
                img = diffusion_model.p_sample(img, t_batch, t)
            
            # Denormalize and convert to numpy
            img_np = ((img.clamp(-1, 1) + 1) / 2).cpu().numpy()
            images.append(img_np)
            
            # Plot images
            for j in range(num_images):
                axes[j, i].imshow(img_np[j, 0], cmap='gray')
                axes[j, i].set_title(f"t={t}")
                axes[j, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'diffusion_process.png'), dpi=300)
    plt.close()
    
    # Create a GIF animation for the first image
    gif_frames = []
    for i in range(len(images)):
        img_pil = Image.fromarray((images[i][0, 0] * 255).astype(np.uint8))
        gif_frames.append(img_pil)
    
    # Save GIF
    gif_path = os.path.join(RESULTS_DIR, 'diffusion_animation.gif')
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=200,
        loop=0
    )
    
    logger.info(f"Diffusion process visualization saved to {RESULTS_DIR}")

def plot_training_loss(losses):
    """
    Plot training loss over epochs.
    
    Creates a line plot of the training loss values over epochs,
    which helps in visualizing the model's convergence during training.
    
    Args:
        losses: List of training losses, one value per epoch
    
    Returns:
        None - Saves the plot to the results directory
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o')
    plt.title('Diffusion Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'))
    plt.close()
    
    logger.info(f"Training loss plot saved to {RESULTS_DIR}")

def run_diffusion_experiment(dataset='mnist', epochs=30, batch_size=128, timesteps=1000):
    """
    Run a complete diffusion model experiment.
    
    Args:
        dataset: Dataset to use ('mnist' or 'fashion_mnist')
        epochs: Number of training epochs
        batch_size: Batch size for training
        timesteps: Number of diffusion timesteps
    
    Returns:
        Dictionary with trained model and results
    """
    # 1. Load data
    logger.info(f"Loading {dataset} dataset...")
    if dataset == 'mnist':
        train_loader = load_mnist_data(batch_size)
    else:
        train_loader = load_fashion_mnist_data(batch_size)
    
    # 2. Initialize model
    logger.info("Initializing diffusion model...")
    model = UNet(in_channels=1, out_channels=1, channel_multipliers=(1, 2, 4)).to(device)
    diffusion = DiffusionModel(model, timesteps=timesteps)
    
    # 3. Train model
    logger.info("Training diffusion model...")
    optimizer = Adam(model.parameters(), lr=2e-4)
    
    losses = diffusion.train(
        dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=epochs,
        save_interval=5,
        sample_interval=5
    )
    
    # 4. Plot training loss
    plot_training_loss(losses)
    
    # 5. Generate samples
    logger.info("Generating samples...")
    samples = diffusion.generate_samples(num_samples=16)
    
    # 6. Visualize diffusion process
    logger.info("Visualizing diffusion process...")
    visualize_diffusion_process(diffusion, num_images=5, steps=10)
    
    # 7. Return results
    return {
        'model': model,
        'diffusion': diffusion,
        'losses': losses,
        'samples': samples
    }

# Example usage
if __name__ == "__main__":
    # For demonstration, use reduced epochs and timesteps
    experiment_results = run_diffusion_experiment(
        dataset='fashion_mnist',     # 'mnist' or 'fashion_mnist'
        epochs=10,                   # Reduced for demonstration
        batch_size=64,               # Smaller batch size
        timesteps=500                # Fewer timesteps
    )
    
    print("\nDiffusion Model Experiment Completed!")
    
    # Access the model and results
    model = experiment_results['model']
    diffusion = experiment_results['diffusion']
    
    # Generate more samples
    print("Generating additional samples...")
    diffusion.generate_samples(num_samples=25)
    
    print(f"All results saved to {RESULTS_DIR}")
""",
                "explanation": "This implementation demonstrates the development of a diffusion model for generative AI, following a progressive denoising approach. The code implements a U-Net architecture with attention mechanisms for the denoising network, handling the forward and reverse diffusion processes systematically. Key components include effective timestep encoding through positional embeddings, customizable sampling strategies, and visualization of the generation process from noise to coherent images. The implementation encompasses the entire workflow from data loading and preprocessing to model training and evaluation, with proper monitoring of training losses and regular checkpointing. Notable features include the detailed visualization of the diffusion process that shows the progressive denoising steps, flexible configuration of diffusion parameters like beta schedules, and modular design that separates the U-Net model from the diffusion process controller. This implementation provides a strong foundation for image generation that could be extended to other domains like text-to-image synthesis, image editing, or 3D generation by adapting the underlying network architecture."
            }
        ],
        
        # PROJECT CONTEXTS (EXACTLY 20 entries)
        "project_contexts": [
            "Predictive maintenance for industrial equipment",
            "Customer churn prediction and retention",
            "Medical image analysis and diagnosis",
            "Natural language processing for sentiment analysis",
            "Recommendation systems for e-commerce platforms",
            "Credit risk assessment and fraud detection",
            "Computer vision for quality control automation",
            "Time series forecasting for inventory management",
            "Personalized content generation with LLMs",
            "Anomaly detection in network security systems",
            "Real-time object detection for autonomous vehicles",
            "Speech recognition and voice assistant development",
            "Energy demand forecasting for smart grids",
            "Customer segmentation for targeted marketing",
            "Automated document processing and information extraction",
            "Chatbot development for customer service",
            "Reinforcement learning for robotic control systems",
            "Pricing optimization using demand prediction",
            "Drug discovery through molecular property prediction",
            "Predictive analytics for healthcare outcomes"
        ],
        
        # ARCHITECTURE PATTERNS (EXACTLY 12 entries)
        "architecture_patterns": [
            {
                "name": "Feature Store",
                "description": "Centralized repository for standardized features used in machine learning models. Feature stores manage the entire lifecycle of features from creation to serving, ensuring consistency between training and inference while enabling reuse across multiple models and teams.",
                "use_cases": ["Enterprise ML platforms", "Real-time ML systems", "Collaborative data science", "Model governance compliance"],
                "pros": ["Reduces feature redundancy", "Ensures training-serving skew prevention", "Enables feature sharing", "Simplifies model deployment", "Provides feature versioning"],
                "cons": ["Initial setup complexity", "Potential performance overhead", "Governance challenges", "Technology lock-in risk", "Learning curve for teams"]
            },
            {
                "name": "Lambda Architecture",
                "description": "Hybrid data processing architecture that combines batch processing for comprehensive, accurate analysis with stream processing for real-time insights. This approach enables both historical analysis and immediate responses to new data, providing a balance between completeness and latency.",
                "use_cases": ["Real-time analytics dashboards", "Fraud detection systems", "IoT data processing", "Recommendation engines"],
                "pros": ["Balances latency and throughput", "Handles both historical and real-time data", "Fault tolerance", "Suitable for complex analytics", "Scalable processing"],
                "cons": ["Code duplication across layers", "Complex maintenance", "Resource intensive", "Consistency challenges", "Increased operational complexity"]
            },
            {
                "name": "Model-as-a-Service",
                "description": "Pattern where machine learning models are deployed as independent services with well-defined APIs, allowing them to be consumed by different applications. This architecture enables separation of model development from application logic and simplifies updating models in production environments.",
                "use_cases": ["Enterprise AI integration", "Third-party model provision", "Cross-platform model sharing", "Multi-language application environments"],
                "pros": ["Technology stack independence", "Independent scaling", "Simplified model updates", "Centralized model management", "Easier A/B testing"],
                "cons": ["Network latency overhead", "Complex monitoring requirements", "Potential API versioning challenges", "Security considerations", "Increased operational overhead"]
            },
            {
                "name": "Microservice ML Architecture",
                "description": "Organization of machine learning components as independently deployable microservices, each responsible for specific functionality like feature extraction, model inference, or post-processing. This approach enables technological diversity, independent scaling, and focused development teams for each component.",
                "use_cases": ["Complex ML pipelines", "Multi-team ML projects", "Heterogeneous model environments", "Gradual legacy system migration"],
                "pros": ["Independent component scaling", "Technology diversity support", "Fault isolation", "Targeted optimization", "Team specialization"],
                "cons": ["Increased operational complexity", "Inter-service communication overhead", "Distributed debugging challenges", "Potential consistency issues", "Resource overhead"]
            },
            {
                "name": "Model Registry",
                "description": "Centralized repository for tracking and managing machine learning models throughout their lifecycle, including versioning, metadata, and deployment status. Model registries provide governance, lineage tracking, and reproducibility while facilitating collaboration and regulatory compliance.",
                "use_cases": ["Enterprise model governance", "Regulatory compliance environments", "Collaborative model development", "Automated deployment pipelines"],
                "pros": ["Model version control", "Deployment tracking", "Metadata management", "Reproducibility support", "Simplified compliance"],
                "cons": ["Additional infrastructure requirements", "Integration complexity", "Potential workflow changes", "Governance overhead", "Learning curve"]
            },
            {
                "name": "Online-Offline Learning",
                "description": "Hybrid approach where models are initially trained offline on historical data and periodically retrained, while continuously updated through online learning mechanisms. This architecture balances comprehensive batch learning with adaptability to new patterns in streaming data.",
                "use_cases": ["Dynamic user behavior modeling", "Fraud detection systems", "Recommendation engines", "Adaptive pricing systems"],
                "pros": ["Adapts to changing patterns", "Reduces retraining frequency", "Balances stability and responsiveness", "Handles concept drift", "Efficient resource utilization"],
                "cons": ["Complex implementation", "Potential model drift", "Difficult monitoring", "Stability concerns", "Increased testing requirements"]
            },
            {
                "name": "Federated Learning",
                "description": "Distributed machine learning approach where models are trained across multiple devices or servers while keeping data localized. This architecture enables learning from decentralized data sources without raw data sharing, addressing privacy concerns and reducing data transfer requirements.",
                "use_cases": ["Mobile device learning", "Healthcare data analysis", "Multi-institution collaboration", "Privacy-sensitive applications"],
                "pros": ["Enhanced data privacy", "Reduced data transfer", "Access to distributed data", "Regulatory compliance", "Edge device utilization"],
                "cons": ["Communication overhead", "Complex implementation", "Potential model bias", "Increased attack surface", "Training convergence challenges"]
            },
            {
                "name": "Data Lake for ML",
                "description": "Centralized repository storing raw, unprocessed data in its native format to support diverse machine learning workloads. This architecture enables exploratory analysis, feature engineering, and model training without predefined schema constraints, allowing flexible data utilization.",
                "use_cases": ["Enterprise data science", "Multi-team ML environments", "Exploratory data analysis", "Historical data mining"],
                "pros": ["Schema flexibility", "Comprehensive data access", "Cost-effective storage", "Support for diverse data types", "Enables data discovery"],
                "cons": ["Data quality challenges", "Governance complexity", "Potential data swamp risk", "Security considerations", "Performance variability"]
            },
            {
                "name": "ML Pipeline Orchestration",
                "description": "Systematic approach to automating and managing end-to-end machine learning workflows, from data ingestion through model deployment and monitoring. This architecture enables reproducibility, automation, and governance of complex multi-stage ML processes while simplifying operational management.",
                "use_cases": ["Production ML systems", "Complex feature engineering", "Regulated ML environments", "Multi-stage model training"],
                "pros": ["Workflow reproducibility", "Automated dependency management", "Simplified monitoring", "Parallelization capabilities", "Version tracking"],
                "cons": ["Initial development overhead", "Potential orchestrator lock-in", "Debugging complexity", "Learning curve", "Infrastructure requirements"]
            },
            {
                "name": "Polyglot ML Architecture",
                "description": "Heterogeneous architecture that leverages multiple programming languages and frameworks to optimize different aspects of the machine learning workflow. This approach enables using specialized tools for their strengths while integrating them into a cohesive system through standardized interfaces.",
                "use_cases": ["Research to production transition", "Performance-critical ML systems", "Specialized algorithm implementation", "Legacy system integration"],
                "pros": ["Optimized component performance", "Best-tool-for-task flexibility", "Specialized expertise utilization", "Incremental adoption", "Technical debt reduction"],
                "cons": ["Integration complexity", "Increased maintenance burden", "Knowledge diversification requirements", "Testing challenges", "Deployment complexity"]
            },
            {
                "name": "Streaming Model Inference",
                "description": "Real-time architecture where machine learning models process continuous data streams to deliver immediate predictions or classifications. This approach enables low-latency decision making on high-velocity data through specialized processing frameworks and optimized model serving infrastructure.",
                "use_cases": ["Real-time fraud detection", "Dynamic pricing systems", "Interactive recommendation systems", "Online advertising"],
                "pros": ["Low-latency predictions", "Immediate value extraction", "Real-time decision support", "Continuous learning capability", "Event-driven architecture compatibility"],
                "cons": ["Complex infrastructure requirements", "Resource intensive", "Model optimization challenges", "Monitoring complexity", "State management difficulties"]
            },
            {
                "name": "Multi-Model Ensemble System",
                "description": "Architecture that combines predictions from multiple independent models to improve accuracy, robustness, and reliability. This approach leverages diverse modeling techniques and perspectives to reduce individual model weaknesses while providing confidence measures through prediction agreement.",
                "use_cases": ["Critical decision systems", "Complex prediction problems", "Uncertainty-aware applications", "High-stakes ML deployments"],
                "pros": ["Improved prediction accuracy", "Reduced overfitting risk", "Model uncertainty estimation", "Robustness to failures", "Diverse approach integration"],
                "cons": ["Increased computational cost", "Complex implementation", "Difficult interpretability", "Multiple model maintenance", "Integration overhead"]
            }
        ],
        
        # DOMAIN RESOURCES (EXACTLY 10 entries)
        "resources": [
            {
                "name": "TensorFlow Documentation",
                "type": "documentation",
                "description": "Comprehensive guides, API references, and tutorials for TensorFlow, covering everything from basic concepts to advanced model development. Includes examples for various neural network architectures and deployment strategies.",
                "url": "https://www.tensorflow.org/docs"
            },
            {
                "name": "PyTorch Documentation",
                "type": "documentation",
                "description": "Official documentation for PyTorch, providing detailed API references, tutorials, and examples for building deep learning models. Covers topics from tensors and autograd to distributed training and model deployment.",
                "url": "https://pytorch.org/docs/stable/index.html"
            },
            {
                "name": "Fast.ai Practical Deep Learning Course",
                "type": "tutorial",
                "description": "Top-down practical approach to deep learning that teaches modern techniques through hands-on projects. The course focuses on best practices and applications rather than theoretical foundations, making it accessible for practitioners.",
                "url": "https://course.fast.ai/"
            },
            {
                "name": "Hugging Face Course",
                "type": "tutorial",
                "description": "Free course teaching natural language processing using the Transformers library. Covers everything from basic transformer concepts to fine-tuning large language models for specific tasks and deployment strategies.",
                "url": "https://huggingface.co/course"
            },
            {
                "name": "Deep Learning by Ian Goodfellow",
                "type": "book",
                "description": "Comprehensive textbook covering the mathematical and conceptual foundations of deep learning. Provides in-depth explanations of neural network architectures, optimization algorithms, and practical methodology for training deep models.",
                "url": "https://www.deeplearningbook.org/"
            },
            {
                "name": "Hands-on Machine Learning with Scikit-Learn and TensorFlow",
                "type": "book",
                "description": "Practical guide to implementing machine learning algorithms using Python libraries. The book balances theory with hands-on examples, covering both traditional ML approaches and deep learning techniques.",
                "url": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/"
            },
            {
                "name": "Kaggle",
                "type": "community",
                "description": "Platform for data science competitions, datasets, notebooks, and discussions. Offers opportunities to practice with real-world problems, learn from shared code, and engage with a community of data scientists and ML engineers.",
                "url": "https://www.kaggle.com/"
            },
            {
                "name": "Stack Overflow Machine Learning",
                "type": "community",
                "description": "Question and answer community focused on machine learning and data science topics. Provides practical solutions to specific implementation challenges and algorithmic problems encountered in ML development.",
                "url": "https://stackoverflow.com/questions/tagged/machine-learning"
            },
            {
                "name": "MLflow",
                "type": "tool",
                "description": "Open-source platform for managing the complete machine learning lifecycle. Provides tracking, packaging, and model registry capabilities to streamline development, reproducibility, and deployment of machine learning models.",
                "url": "https://mlflow.org/"
            },
            {
                "name": "Weights & Biases",
                "type": "tool",
                "description": "Machine learning experiment tracking, dataset versioning, and model management platform. Features visualization tools for monitoring training progress, comparing experiments, and collaborating with team members on ML projects.",
                "url": "https://wandb.ai/"
            }
        ]
    }
