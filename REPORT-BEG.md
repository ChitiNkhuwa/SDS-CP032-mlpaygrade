# 📄 MLPayGrade – Project Report - 🟢 **Beginner Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Week 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: What roles or experience levels yield the highest average salary?

### 🔑 Question 2: Does remote work correlate with higher or lower salaries?

### 🔑 Question 3: Are there differences in salary based on company size or location?

### 🔑 Question 4: How consistent are salaries across similar job titles?

---

## ✅ Week 2: Feature Engineering & Data Preprocessing

### 🔑 Question 1:
**Can you create any new features from the existing dataset that might improve model performance? Why might these features help?**

💡 **Hint:**  
Think about interaction features (e.g., experience level + remote ratio).  
Consider simplifying complex categories or combining related ones.  
Try binning numerical features or creating flags for rare categories.  
Ask: “What kind of information would help a model make a better salary prediction?”

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What transformations or encodings are necessary for the categorical variables, and what encoding method is most appropriate for each one?**

💡 **Hint:**  
Use `.nunique()` and `.value_counts()` to see cardinality and frequency.  
Low-cardinality → One-hot encoding  
High-cardinality (e.g., job title, company location) → Consider target encoding or frequency encoding  
Visualize the distribution of values with bar plots before deciding.  
Think: “Does the order of categories matter?” If yes → ordinal. If not → one-hot or target.

✏️ *Your answer here...*

---

### 🔑 Question 3:
**What baseline model can you start with, and what performance can you reasonably expect?**

💡 **Hint:**  
Start with Linear Regression as your baseline model.  
Split your data into train/test using `train_test_split`.  
Use metrics like RMSE, MAE, and R² to evaluate.  
Don't expect high accuracy here — the goal is to understand limitations and build a benchmark.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**How would you explain the difference between underfitting and overfitting in the context of your baseline model?**

💡 **Hint:**  
Underfitting → Model performs poorly on both train and test sets.  
Overfitting → Model performs well on train but poorly on test.  
Compare train vs. test errors.  
Visualize residual plots — if residuals are random and centered around 0, that’s good.

✏️ *Your answer here...*

---

### 🔑 Question 5:
**Are there any features that should be removed or transformed before modeling, and why?**

💡 **Hint:**  
Look for redundant columns (e.g., is `job_title` too similar to `experience_level`?)  
Check for skewed features using histograms (e.g., `salary_in_usd`) and consider log-transforming them.  
Evaluate multicollinearity using pairplots or VIF (Variance Inflation Factor).  
Consider dropping or combining rare categories in high-cardinality categorical variables.

✏️ *Your answer here...*

---


---

## ✅ Week 3: Model Development & Experimentation

### 🔑 Question 1:
**What models did you train to predict salary, and what assumptions or strengths does each model bring to this problem?**  
🎯 *Purpose: Tests ability to match algorithms with problem characteristics and justify choices.*

💡 **Hint:**  
Discuss Linear Regression (baseline, interpretable), Random Forest (handles non-linearity, less preprocessing), and XGBoost (regularized boosting, high performance).  
Include the modeling code used for training.

✏️ *Your answer here...*

---

### 🔑 Question 2:
**What were the RMSE, MAE, and R² scores for each model on the test set? What patterns do you notice in how the models performed?**  
🎯 *Purpose: Tests metric interpretation and comparative analysis skills.*

💡 **Hint:**  
Present your evaluation results in a table or chart.  
Explain which model performed best and why that might be (e.g., feature interactions, overfitting resilience).  
Use `sklearn.metrics` functions to compute the metrics.

✏️ *Your answer here...*

---

### 🔑 Question 3:
**How do the training and test errors compare for each model? Do any models show signs of overfitting or underfitting?**  
🎯 *Purpose: Tests understanding of generalization and error gaps.*

💡 **Hint:**  
If training error is much lower than test → overfitting  
If both errors are high → underfitting  
Use error values or visualize residuals to explain your reasoning.

✏️ *Your answer here...*

---

### 🔑 Question 4:
**Which features were most important to your best-performing model, and how did you determine that? Do the top features make sense based on domain intuition?**  
🎯 *Purpose: Tests model interpretation and connection to domain knowledge.*

💡 **Hint:**  
Use `.feature_importances_` (for Random Forest/XGBoost) or `.coef_` (for Linear Regression).  
Visualize feature importance as a bar chart.  
Explain whether the top features align with what you'd expect in real life (e.g., job title, experience level, etc.).

✏️ *Your answer here...*

---

### 🔑 Question 5:
**How did you use MLflow to track your experiments, and what did it help you learn or compare more effectively?**  
🎯 *Purpose: Tests experiment tracking habits and insights gained from tracking.*

💡 **Hint:**  
Log metrics, model type, and parameters for each run.  
Did MLflow help you compare model versions, detect overfitting, or identify the best configuration?  
Include screenshots or MLflow logs if possible.

✏️ *Your answer here...*

---

## ✅ Week 4: Model Selection & Hyperparameter Tuning

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

## ✅ Week 5: Model Deployment

> Document your approach to building and deploying the Streamlit app, including design decisions, deployment steps, and challenges.

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

## ✨ Final Reflections

> What did you learn from this project? What would you do differently next time? What did AI tools help you with the most?

✏️ *Your final thoughts here...*

---
