# titanic-ml-competition
Kaggle Titanic ML Competition (built with pandas, sklearn, logistic regression)

This repository contains my solution for the **[Kaggle Titanic – Machine Learning from Disaster](https://www.kaggle.com/c/titanic)** challenge.  
The goal of the project is to predict passenger survival using basic machine-learning techniques in Python.

---

## Overview
This was one of my first end-to-end data-science projects.  
It demonstrates the full workflow of a supervised-learning task:

1. Loading and cleaning tabular data with **pandas**
2. Encoding categorical features and handling missing values
3. Training a **Logistic Regression** model with **scikit-learn**
4. Evaluating model accuracy on a validation set
5. Exporting predictions for Kaggle submission

---

## Project Files
| File | Description |
|------|--------------|
| `titanic-model.py` | Main Python script used to train the model and generate predictions |
| `submission.csv` | Output file submitted to Kaggle |
| `README.md` | Project documentation |

---

## Key Learnings
- Data preprocessing and feature handling in pandas  
- Avoiding under- and over-fitting through proper train/validation splits  
- Interpreting model metrics such as accuracy  
- Submitting predictions to Kaggle competitions  

---

## Results
The final Logistic Regression model achieved a Kaggle leaderboard score of **`0.78229`**, placing me at 2344th place out of the 14403 who have attempted the challenge.  
While simple, the project gave me a strong foundation in model evaluation and reproducible experimentation.

---

## Acknowledgements
- Data provided by **Kaggle Titanic Competition**  
- Libraries used: `pandas`, `numpy`, `scikit-learn`

---

This repository is part of my personal learning portfolio.  
It marks my first complete end-to-end machine-learning pipeline and serves as a baseline for future, more advanced projects.
