# Breast Cancer Detection

## Overview

This project implements a machine learning model to detect breast cancer using classification techniques. The goal is to accurately classify tumors as malignant or benign based on various input features. The project includes both a trained model and a FastAPI-based REST API for making predictions.

## Dataset

The model utilizes the [Breast Cancer Wisconsin (Diagnostic) Data Set][1], which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. Each instance is labeled as either malignant or benign. The dataset includes 569 samples with 30 features each.

## Model Architecture

After extensive model comparison using k-fold cross-validation, a Logistic Regression model was chosen as it achieved the highest accuracy (97.6%). The model pipeline includes:

1. **Feature Preprocessing**: MinMaxScaler for feature normalization
2. **Dimensionality Reduction**: PCA (Principal Component Analysis)
3. **Classification**: Logistic Regression with the following parameters:
   - **Solver**: `'liblinear'` – suitable for small datasets
   - **Class Weight**: `{0: 1.684, 1: 1.0}` – addresses class imbalance
   - **Random State**: `42` – ensures reproducibility

## Performance Metrics

The model achieves excellent performance on the test set:
- Accuracy: 96%
- Precision: 96%
- Recall: 96%
- F1-Score: 96%
- Error Rate: 4.386%

## Project Structure

```
├── app.py                  # FastAPI application
├── data/                   # Data directory
├── models/                 # Trained model files
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## Installation & Requirements

1. Clone the repository:
```bash
git clone https://github.com/isaacmenchaca97/breast_cancer_detection.git
cd breast_cancer_detection
```

2. Create and activate a virtual environment:
```bash
make create_environment
```

3. Install dependencies:
```bash
make requirements
```

## Usage

### As a Python Package

```python
import pickle
import numpy as np

# Load the model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
features = np.array([...])  # Your input features
prediction = model.predict(features)
```

### As a REST API

1. Start the FastAPI server:
```bash
uvicorn app:app --reload
```

2. Make predictions via HTTP:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [...]}'
```

## Development

The project uses several development tools:
- **MLflow**: For experiment tracking and model versioning
- **pytest**: For unit testing
- **ruff**: For code linting and formatting

Run tests:
```bash
make test
```

## Results

The model demonstrates robust performance in distinguishing between malignant and benign tumors:

- High accuracy across all metrics (precision, recall, F1-score)
- Strong ROC-AUC performance
- Low misclassification rate (4.386%)

## Future Improvements

- Implement additional preprocessing techniques
- Explore deep learning approaches
- Add model explainability features
- Enhance API documentation
- Implement batch prediction endpoints

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Principal Component Analysis (PCA) implementation is based on the work of [Daksh Bhatnagar][2]
- Dataset provided by the University of Wisconsin

[1]: https://doi.org/10.24432/C5DW2B
[2]: https://www.kaggle.com/code/bhatnagardaksh/pca-and-lda-implementation/notebook 