# Breast Cancer Detection

## Overview

This project implements a machine learning model to detect breast cancer using classification techniques. The goal is to accurately classify tumors as malignant or benign based on various input features.

## Dataset

The model utilizes the [Breast Cancer Wisconsin (Diagnostic) Data Set][1], which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. Each instance is labeled as either malignant or benign.

## Model Architecture

A Logistic Regression model is employed with the following parameters:

- **Solver**: `'liblinear'` – suitable for small datasets and supports L1 and L2 regularization.
- **Class Weight**: `{0: 1.684, 1: 1.0}` – assigns a higher weight to the benign class to address class imbalance.
- **Random State**: `42` – ensures reproducibility of results.

## Installation & Requirements

To run this project, ensure you have the following installed:

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

## Usage

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/isaacmenchaca97/breast_cancer_detection.git
cd breast_cancer_detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Jupyter Notebook to explore the analysis and model implementation:

```bash
jupyter notebook breast_cancer.ipynb
```

## Results

The model achieved an accuracy of 96% on the test set, demonstrating its effectiveness in distinguishing between malignant and benign tumors.

![Screenshot 2025-03-22 at 11 37 19 p m](https://github.com/user-attachments/assets/53f1ada1-4fd8-4371-bbd9-aacc57f5a753)
![Screenshot 2025-03-22 at 11 37 32 p m](https://github.com/user-attachments/assets/36bae02d-3011-4ed7-8c62-2865a4a79e61)


## Future Improvements

- Implement additional preprocessing techniques to enhance feature extraction.
- Explore other classification algorithms to potentially improve performance.

## License

This project is licensed under the MIT License.

## Acknowledgments

The Principal Component Analysis (PCA) implementation in this project is based on the work of [Daksh Bhatnagar][2].

[1]: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://doi.org/10.24432/C5DW2B)
[2]: https://www.kaggle.com/code/bhatnagardaksh/pca-and-lda-implementation/notebook 