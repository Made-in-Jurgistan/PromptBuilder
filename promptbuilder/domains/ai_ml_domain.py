"""
ai_ml_domain.py

This module defines the AI/ML domain metadata, languages, frameworks, technologies, concepts, best practices, and common problems for PromptBuilder, strictly following the domain_template.py structure. It contains only static, structured domain knowledge and configuration—no code examples or executable logic.
"""

from typing import Dict, Any

def get_ai_ml_domain() -> Dict[str, Any]:
    """Return the AI/ML domain metadata and mappings (no code examples)."""
    return {
        # DOMAIN METADATA
        "name": "AI/ML",
        "description": "Development and application of artificial intelligence and machine learning systems for predictive modeling, natural language processing, computer vision, and other intelligent systems. This domain encompasses both theoretical foundations and practical implementations that enable machines to learn from data and make decisions.",
        # LANGUAGES (EXACTLY 6 entries)
        "languages": [
            {
                "name": "Python",
                "version": "3.12",
                "description": "Dominant language for AI/ML with the richest ecosystem for deep learning, LLMs, and data science. Excels in rapid prototyping, research, and production APIs.",
                "popularity": "high",
                "typical_uses": ["LLMs", "Deep learning", "Data science", "Production APIs"]
            },
            {
                "name": "Julia",
                "version": "1.11",
                "description": "High-performance language for numerical and scientific computing. Excels at differentiable programming, ML research, and high-speed model training.",
                "popularity": "medium",
                "typical_uses": ["Scientific ML", "Differentiable programming", "Numerical optimization", "Research"]
            },
            {
                "name": "Rust",
                "version": "1.78",
                "description": "Memory-safe, high-performance systems language used for ML infrastructure, inference engines, and fast data pipelines.",
                "popularity": "medium",
                "typical_uses": ["Inference engines", "ML infrastructure", "Data pipelines", "Performance-critical systems"]
            },
            {
                "name": "Go",
                "version": "1.22",
                "description": "Efficient, statically typed language for scalable ML microservices, distributed systems, and production data engineering.",
                "popularity": "medium",
                "typical_uses": ["ML microservices", "Distributed systems", "Data engineering", "Production APIs"]
            },
            {
                "name": "R",
                "version": "4.4",
                "description": "Popular for statistical analysis, visualization, and specialized ML in academia and healthcare. Excels at rapid prototyping and statistical modeling.",
                "popularity": "low",
                "typical_uses": ["Statistical modeling", "Visualization", "Healthcare analytics", "Academic research"]
            },
            {
                "name": "C++",
                "version": "23",
                "description": "Core language for high-performance ML libraries, inference engines, and custom CUDA kernels. Used for production-grade, latency-critical ML systems.",
                "popularity": "medium",
                "typical_uses": ["ML library development", "Inference engines", "CUDA kernels", "Embedded AI"]
            }
        ],
        # FRAMEWORKS (EXACTLY 8 entries)
        "frameworks": [
            {
                "name": "PyTorch",
                "version": "2.3",
                "description": "Leading deep learning framework for research and production, supporting LLMs, vision, and multimodal AI. Excels in flexibility, dynamic computation graphs, and community support.",
                "language": "Python/C++",
                "typical_uses": ["LLMs", "Deep learning", "Vision", "Production APIs"],
                "learning_curve": "moderate",
                "category": "Deep Learning"
            },
            {
                "name": "JAX",
                "version": "0.4.25",
                "description": "High-performance ML framework for composable function transformations, LLMs, and generative AI. Excels at automatic differentiation and hardware acceleration.",
                "language": "Python",
                "typical_uses": ["LLMs", "Generative AI", "Differentiable programming", "Research"],
                "learning_curve": "steep",
                "category": "Deep Learning"
            },
            {
                "name": "LangChain",
                "version": "0.3",
                "description": "Framework for building LLM-powered applications, RAG pipelines, and agentic workflows. Excels at tool integration and orchestration for LLMOps.",
                "language": "Python/JavaScript",
                "typical_uses": ["RAG", "LLM apps", "Agents", "Prompt engineering"],
                "learning_curve": "gentle",
                "category": "LLM Application"
            },
            {
                "name": "LlamaIndex",
                "version": "0.11",
                "description": "Data framework for connecting LLMs to external data and advanced retrieval. Excels at RAG and enterprise LLM pipelines.",
                "language": "Python",
                "typical_uses": ["RAG", "Enterprise LLM apps", "Data connectors", "Indexing"],
                "learning_curve": "gentle",
                "category": "LLM Application"
            },
            {
                "name": "Hugging Face Transformers",
                "version": "4.43",
                "description": "De facto library for LLMs, NLP, vision, and multimodal models. Excels in model availability, transfer learning, and rapid prototyping.",
                "language": "Python",
                "typical_uses": ["LLMs", "NLP", "Vision", "Transfer learning"],
                "learning_curve": "moderate",
                "category": "LLM/NLP"
            },
            {
                "name": "DeepSpeed",
                "version": "0.15",
                "description": "Distributed training and inference optimization library for large LLMs and generative models. Excels at memory efficiency and large-scale model support.",
                "language": "Python",
                "typical_uses": ["Distributed training", "LLMs", "Inference optimization", "Memory efficiency"],
                "learning_curve": "steep",
                "category": "Distributed Training"
            },
            {
                "name": "vLLM",
                "version": "0.5",
                "description": "High-throughput, low-latency inference engine for LLMs, supporting advanced tensor parallelism and paged attention. Excels at production LLM inference.",
                "language": "Python/C++",
                "typical_uses": ["LLM inference", "Serving", "Production APIs", "Batch inference"],
                "learning_curve": "moderate",
                "category": "Inference Optimization"
            },
            {
                "name": "scikit-learn",
                "version": "1.6",
                "description": "Classic ML library for Python, providing robust implementations of standard algorithms and preprocessing tools. Excels at tabular data and rapid prototyping.",
                "language": "Python",
                "typical_uses": ["Classical ML", "Tabular data", "Prototyping", "Feature engineering"],
                "learning_curve": "gentle",
                "category": "Machine Learning"
            }
        ],
        # TECHNOLOGIES (EXACTLY 8 entries)
        "technologies": [
            {
                "name": "Docker",
                "version": "26.0",
                "description": "Containerization platform enabling reproducible, portable ML environments and scalable deployment of AI/ML services. Integrates with orchestration tools for robust production workflows.",
                "category": "Containerization",
                "maturity": "mature"
            },
            {
                "name": "Kubernetes",
                "version": "1.30",
                "description": "Container orchestration system for automating deployment, scaling, and management of ML workloads in production. Supports distributed training and model serving at scale.",
                "category": "Orchestration",
                "maturity": "mature"
            },
            {
                "name": "Ray",
                "version": "2.9",
                "description": "Distributed computing framework for scaling Python and ML workloads, with libraries for hyperparameter tuning, reinforcement learning, and distributed data processing.",
                "category": "Distributed Computing",
                "maturity": "established"
            },
            {
                "name": "MLflow",
                "version": "2.8",
                "description": "Open-source platform for managing the ML lifecycle, including experiment tracking, model registry, and reproducible deployment. Integrates with major ML frameworks and MLOps stacks.",
                "category": "MLOps",
                "maturity": "established"
            },
            {
                "name": "Weights & Biases",
                "version": "0.18",
                "description": "Experiment tracking and collaboration platform for ML teams, supporting large-scale LLM and generative AI workflows. Enables visualization, comparison, and sharing of results.",
                "category": "Experiment Tracking",
                "maturity": "established"
            },
            {
                "name": "ONNX",
                "version": "1.19",
                "description": "Open Neural Network Exchange format for cross-framework model interoperability and deployment. Facilitates model export and optimization for diverse hardware.",
                "category": "Model Interoperability",
                "maturity": "established"
            },
            {
                "name": "NVIDIA CUDA",
                "version": "12.5",
                "description": "GPU computing platform for accelerating deep learning and LLM training/inference. Provides APIs and libraries for high-performance ML workloads.",
                "category": "Hardware Acceleration",
                "maturity": "mature"
            },
            {
                "name": "Apache Airflow",
                "version": "2.10",
                "description": "Workflow orchestration platform for automating and monitoring ML pipelines and data engineering tasks. Enables scheduling, dependency management, and observability for production ML workflows.",
                "category": "Workflow Orchestration",
                "maturity": "established"
            }
        ],
        # CONCEPTS (EXACTLY 50 entries)
        "concepts": [
            "Activation Functions",
            "Active Learning",
            "Agentic Workflows",
            "Attention Mechanisms",
            "Autoencoder",
            "AutoML",
            "Backpropagation",
            "Batch Normalization",
            "Bias-Variance Tradeoff",
            "Convolutional Neural Networks",
            "Cross-Validation",
            "Data Augmentation",
            "Data Versioning",
            "Deep Learning",
            "Dimensionality Reduction",
            "Embeddings",
            "Ensemble Methods",
            "Explainable AI",
            "Feature Engineering",
            "Feature Selection",
            "Few-Shot Learning",
            "Foundation Models",
            "Federated Learning",
            "GANs",
            "Gradient Descent",
            "Hyperparameter Tuning",
            "Imbalanced Learning",
            "Knowledge Distillation",
            "Large Language Models",
            "Model Compression",
            "Model Drift",
            "Model Interpretability",
            "Model Quantization",
            "MLOps",
            "Natural Language Processing",
            "Neural Architecture Search",
            "Neural Networks",
            "Online Learning",
            "Overfitting and Underfitting",
            "Parameter-Efficient Fine-Tuning",
            "Precision and Recall",
            "Prompt Engineering",
            "Prompt Injection",
            "Reinforcement Learning",
            "Recurrent Neural Networks",
            "Regularization",
            "Responsible AI",
            "Retrieval-Augmented Generation",
            "ROC Curve and AUC",
            "Self-Supervised Learning",
            "Semi-Supervised Learning",
            "Synthetic Data Generation"
        ],
        # BEST PRACTICES (EXACTLY 10 entries)
        "best_practices": [
            {
                "name": "Responsible and Ethical AI",
                "description": "Develop ML systems with fairness, transparency, and accountability. Use bias detection, explainability, and privacy tools to ensure ethical outcomes and regulatory compliance.",
                "benefits": ["Fair outcomes", "Regulatory compliance", "User trust", "Reduced bias"],
                "related_concepts": ["Responsible AI", "Explainable AI", "Model Interpretability", "Bias-Variance Tradeoff"]
            },
            {
                "name": "MLOps and CI/CD",
                "description": "Automate the ML lifecycle with robust CI/CD, reproducible pipelines, and continuous monitoring. Use MLOps stacks for scalable, reliable production workflows.",
                "benefits": ["Reproducibility", "Scalability", "Faster deployment", "Reduced errors"],
                "related_concepts": ["MLOps", "Model Drift", "Model Monitoring", "Cross-Validation"]
            },
            {
                "name": "Prompt Engineering",
                "description": "Design, test, and optimize prompts for LLMs and generative AI. Use prompt libraries and evaluation frameworks for robust LLM applications.",
                "benefits": ["Better LLM outputs", "Reduced hallucination", "Faster iteration", "Improved reliability"],
                "related_concepts": ["Prompt Engineering", "Large Language Models", "Retrieval-Augmented Generation", "Prompt Injection"]
            },
            {
                "name": "Experiment Tracking",
                "description": "Systematically record all experimental parameters, datasets, and results to ensure reproducibility and facilitate analysis. Use experiment tracking tools for collaboration.",
                "benefits": ["Reproducibility", "Collaboration", "Traceability", "Faster iteration"],
                "related_concepts": ["Experiment Tracking", "Data Versioning", "Model Versioning", "Hyperparameter Tuning"]
            },
            {
                "name": "Data Validation and Cleaning",
                "description": "Validate and clean data before model training to address missing values, outliers, and inconsistencies. Use automated data quality checks for robust models.",
                "benefits": ["Improved accuracy", "Reduced errors", "Cleaner data", "Fewer failures"],
                "related_concepts": ["Data Augmentation", "Feature Engineering", "Imbalanced Learning", "Data Versioning"]
            },
            {
                "name": "Monitoring and Observability",
                "description": "Continuously monitor models and data for performance, drift, and errors. Use observability stacks for real-time alerts and dashboards.",
                "benefits": ["Early issue detection", "Stable performance", "Faster recovery", "Reduced downtime"],
                "related_concepts": ["Model Drift", "Model Monitoring", "Drift Detection", "Overfitting and Underfitting"]
            },
            {
                "name": "Automated Testing",
                "description": "Use automated tests to validate code, data, and model behavior. Include unit, integration, and model validation tests for robust ML systems.",
                "benefits": ["Fewer bugs", "Higher quality", "Faster releases", "Reliable systems"],
                "related_concepts": ["Testing", "Cross-Validation", "Feature Engineering", "Regularization"]
            },
            {
                "name": "Cross-Validation",
                "description": "Evaluate models on multiple data splits to ensure generalization and stability. Use robust metrics and stratified sampling for reliable evaluation.",
                "benefits": ["Better generalization", "Reduced overfitting", "Reliable metrics", "Improved trust"],
                "related_concepts": ["Cross-Validation", "Overfitting and Underfitting", "Bias-Variance Tradeoff", "Model Evaluation"]
            },
            {
                "name": "Feature Selection",
                "description": "Identify and use the most informative features for model training to improve performance and reduce overfitting. Use automated and manual selection methods.",
                "benefits": ["Improved accuracy", "Reduced complexity", "Faster training", "Lower cost"],
                "related_concepts": ["Feature Engineering", "Dimensionality Reduction", "Regularization", "Model Compression"]
            },
            {
                "name": "Performance Optimization",
                "description": "Continuously analyze and optimize the performance and cost of models and systems. Use profiling, quantization, and hardware acceleration for efficient ML workflows.",
                "benefits": ["Lower latency", "Reduced cost", "Faster inference", "Efficient scaling"],
                "related_concepts": ["Inference Optimization", "Model Quantization", "Hardware Acceleration", "Batch Normalization"]
            }
        ],
        # COMMON PROBLEMS (EXACTLY 10 entries)
        "common_problems": [
            {
                "name": "Overfitting",
                "description": "Model performs well on training data but poorly on unseen data due to learning noise or peculiarities in the training set.",
                "indicators": ["High training accuracy", "Low validation accuracy", "Complex model", "Poor generalization"],
                "causes": ["Too complex model", "Insufficient data", "Noisy features", "Lack of regularization", "Overtraining"],
                "solutions": ["Regularization", "More data", "Simpler model", "Cross-validation", "Early stopping"]
            },
            {
                "name": "Data Leakage",
                "description": "Training data includes information not available during inference, leading to unrealistically high performance metrics during development but failure in production.",
                "indicators": ["Unrealistic performance", "Production collapse", "Sudden drop in accuracy", "Unexpected results"],
                "causes": ["Target-derived features", "Preprocessing before split", "Improper data handling", "Feature engineering errors", "Temporal leakage"],
                "solutions": ["Proper splits", "Pipeline preprocessing", "Time-based validation", "Feature audit", "Strict separation"]
            },
            {
                "name": "Training-Serving Skew",
                "description": "Difference between training and production environments causes model performance degradation when deployed.",
                "indicators": ["Good offline metrics", "Poor online performance", "Unexpected drift", "Inconsistent results"],
                "causes": ["Different preprocessing", "Data drift", "Feature mismatch", "Environment differences", "Code divergence"],
                "solutions": ["End-to-end testing", "Feature monitoring", "Unified preprocessing", "Environment parity", "Automated deployment"]
            },
            {
                "name": "Model Drift",
                "description": "Model performance deteriorates over time as data patterns change, requiring ongoing attention to maintain effectiveness.",
                "indicators": ["Declining metrics", "Increasing errors", "Frequent retraining", "Performance decay"],
                "causes": ["Changing data", "User behavior shift", "Seasonal effects", "External factors", "Feature drift"],
                "solutions": ["Regular retraining", "Drift detection", "Continuous monitoring", "Adaptive models", "Alerting"]
            },
            {
                "name": "GPU Memory Errors",
                "description": "Out of memory errors when training deep learning models on GPUs, preventing successful completion of training runs.",
                "indicators": ["CUDA OOM errors", "Training crashes", "Batch size limits", "Resource exhaustion"],
                "causes": ["Large batch size", "Complex model", "Memory leaks", "Inefficient code", "Insufficient hardware"],
                "solutions": ["Gradient accumulation", "Mixed precision", "Model parallelism", "Reduce batch size", "Optimize code"]
            },
            {
                "name": "Imbalanced Data",
                "description": "Class distribution in training data is highly skewed, leading to models biased toward majority classes and poor minority class performance.",
                "indicators": ["High accuracy, low recall", "Minority class errors", "Biased predictions", "Poor F1 score"],
                "causes": ["Natural imbalance", "Sampling bias", "Data collection issues", "Labeling errors", "Small minority class"],
                "solutions": ["Resampling", "Class weighting", "SMOTE", "Data augmentation", "Custom loss"]
            },
            {
                "name": "Slow Inference",
                "description": "Model takes too long to make predictions in production, creating bottlenecks or increased operational costs.",
                "indicators": ["High latency", "Timeouts", "Slow response", "User complaints"],
                "causes": ["Model complexity", "Inefficient code", "Large input size", "Resource contention", "Unoptimized hardware"],
                "solutions": ["Quantization", "Distillation", "Pruning", "Batching", "Hardware acceleration"]
            },
            {
                "name": "Reproducibility Issues",
                "description": "Inability to consistently reproduce model training results due to randomness, environment variations, or insufficient tracking of parameters and dependencies.",
                "indicators": ["Different results", "Inconsistent metrics", "Unstable experiments", "Difficult debugging"],
                "causes": ["Random seed not fixed", "Version differences", "Untracked dependencies", "Non-deterministic code", "Environment drift"],
                "solutions": ["Fixed seeds", "Containerization", "Version pinning", "Experiment tracking", "Reproducible pipelines"]
            },
            {
                "name": "Feature Engineering Complexity",
                "description": "Difficulty in identifying, creating, and selecting the most relevant features for a given problem, requiring significant domain expertise and experimentation.",
                "indicators": ["Labor-intensive preprocessing", "Slow iteration", "Feature bloat", "Unclear importance"],
                "causes": ["Domain complexity", "Data heterogeneity", "Lack of automation", "Manual feature creation", "Changing requirements"],
                "solutions": ["Automated feature generation", "Representation learning", "Feature selection", "Domain collaboration", "Feature stores"]
            },
            {
                "name": "Cold Start Problem",
                "description": "Challenge of making accurate predictions for new users or items with little to no historical data, especially in recommendation systems.",
                "indicators": ["Poor recommendations", "Low engagement", "Sparse data", "Unreliable predictions"],
                "causes": ["No interaction history", "Popularity bias", "Sparse features", "New item/user", "Limited context"],
                "solutions": ["Hybrid approaches", "Content-based features", "Transfer learning", "Synthetic data", "Active learning"]
            }
        ],
        # RESOURCES (EXACTLY 10 entries)
        "resources": [
            {
                "name": "Papers With Code",
                "type": "tool",
                "description": "Latest ML papers, benchmarks, and code implementations. Essential for tracking state-of-the-art methods and reproducible research.",
                "url": "https://paperswithcode.com/"
            },
            {
                "name": "Hugging Face Hub",
                "type": "tool",
                "description": "Central repository for open-source models, datasets, and LLMs. Enables rapid prototyping and sharing of ML assets.",
                "url": "https://huggingface.co/"
            },
            {
                "name": "arXiv.org AI/ML",
                "type": "documentation",
                "description": "Preprints of the latest research in AI and ML. Valuable for staying current with foundational and emerging topics.",
                "url": "https://arxiv.org/list/cs.AI/recent"
            },
            {
                "name": "DeepLearning.AI",
                "type": "tutorial",
                "description": "Courses, news, and resources for deep learning and LLMs. Useful for structured learning and upskilling.",
                "url": "https://www.deeplearning.ai/"
            },
            {
                "name": "OpenAI Cookbook",
                "type": "documentation",
                "description": "Best practices, recipes, and guides for LLMs and generative AI. Practical resource for production LLM workflows.",
                "url": "https://cookbook.openai.com/"
            },
            {
                "name": "Google AI Blog",
                "type": "community",
                "description": "Research updates and best practices from Google AI. Useful for trends, case studies, and applied research.",
                "url": "https://ai.googleblog.com/"
            },
            {
                "name": "MLflow Documentation",
                "type": "documentation",
                "description": "Comprehensive documentation for MLflow and MLOps. Covers experiment tracking, model registry, and deployment.",
                "url": "https://mlflow.org/docs/latest/index.html"
            },
            {
                "name": "LangChain Docs",
                "type": "documentation",
                "description": "Official documentation for LangChain LLM application framework. Covers RAG, agentic workflows, and tool integration.",
                "url": "https://python.langchain.com/docs/"
            },
            {
                "name": "vLLM Docs",
                "type": "documentation",
                "description": "Documentation for vLLM, the high-throughput LLM inference engine. Useful for production LLM serving and optimization.",
                "url": "https://vllm.readthedocs.io/en/latest/"
            },
            {
                "name": "Evidently AI",
                "type": "tool",
                "description": "Open-source tools for ML monitoring, data drift, and model evaluation. Essential for production monitoring and observability.",
                "url": "https://evidentlyai.com/"
            }
        ]
    }