# 🔍 Lead Conversion Prediction - Artefact

This repository demonstrates a machine learning pipeline built to predict lead conversion using historical CRM data from `archive.csv`. The project showcases two supervised models: Random Forest and Logistic Regression, both implemented in Python via Jupyter Notebook.

---

## Dataset Description

The dataset contains lead information from a CRM system with attributes such as:

- Lead Source, Lead Origin, Last Activity
- Web interaction data**: Total Visits, Page Views per Visit, Time on Website
- Lead demographics: Country, City, Specialization, Occupation
- Target variable: `Converted` (1 = converted, 0 = not converted)

All preprocessing and feature engineering is performed dynamically within the notebook.

---

##  Project Workflow

### Data Preprocessing

- Dropped non-predictive identifiers (`Prospect ID`, `Lead Number`)
- Replaced missing values (dropped rows with >30% NaNs, others filled as "Unknown")
- Encoded categorical variables using `LabelEncoder`

### Model 1: Random Forest Classifier

A tree-based ensemble model used for its robustness and interpretability.

#### Steps:
- Trained on 80% of the data, tested on 20%
- Evaluated using accuracy score and classification report
- Visualized top 10 most important features using feature importance plot

### Model 2: Logistic Regression

A linear model for binary classification offering interpretability of feature influence.

#### Steps:
- Trained with `max_iter=1000`
- Evaluated using classification report and confusion matrix
- Top 10 most influential features visualized using model coefficients
- Predictions displayed for the first 10 records

### Visual Insights

- **Boxplot** showing correlation between `Total Time Spent on Website` and conversion rate
- **Confusion matrix** showing true/false positives and negatives for model evaluation

---

## Key Outputs

- **Accuracy** of both models (varies by dataset, typically around 70–80%)
- **Top predictors** of lead conversion include:
  - Last Activity
  - Total Time Spent on Website
  - Lead Source
  - Asymmetrique Profile Index

---

## Dependencies

Make sure to install the following packages:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
