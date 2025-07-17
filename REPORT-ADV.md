# 📄 MLPayGrade – Project Report - 🔴 **Advanced Track**

Welcome to your personal project report!  
Use this file to answer the key reflection questions for each phase of the project. This report is designed to help you think like a data scientist, guide AI tools more effectively, and prepare for real-world job interviews.

---

## ✅ Phase 1: Setup & Exploratory Data Analysis (EDA)

> Answer the EDA questions provided in the project materials here. Focus on data quality, trends, anomalies, and relationships.

### 🔑 Question 1: What roles or experience levels yield the highest average salary?

### 🔑 Question 2: Does remote work correlate with higher or lower salaries?

### 🔑 Question 3: Are there differences in salary based on company size or location?

### 🔑 Question 4: How consistent are salaries across similar job titles?

---

## ✅ Phase 2: Model Development

> This phase spans 3 weeks. Answer each set of questions weekly as you build, train, evaluate, and improve your models.

---

### 📆 Week 1: Feature Engineering & Data Preprocessing

#### 🔑 Question 1:
**Which categorical features are high-cardinality, and how will you encode them for use with embedding layers?**  

💡 **Hint:**  
Use `.nunique()` and `.value_counts()` to inspect cardinality.  
Use `LabelEncoder` or map categories to integer IDs.  
Think about issues like rare categories, overfitting, and embedding size selection.

✏️ *Your answer here...*

---

#### 🔑 Question 2:
**What numerical features in the dataset needed to be scaled before being input into a neural network, and what scaling method did you choose?**  

💡 **Hint:**  
Use `df.describe()` and histograms to evaluate spread and skew.  
Neural networks are sensitive to feature scale.  
Choose between `StandardScaler`, `MinMaxScaler`, or log-transform based on the data.

✏️ *Your answer here...*

---

#### 🔑 Question 3:
**Did you create any new features based on domain knowledge or data relationships? If yes, what are they and why might they help the model predict salary more accurately?**  

💡 **Hint:**  
Try combining features like `job_title + experience_level`, or flags like `is_remote = remote_ratio == 100`.  
Think like a recruiter: What drives salary changes?

✏️ *Your answer here...*

---

#### 🔑 Question 4:
**Which features, if any, did you choose to drop or simplify before modeling, and what was your justification?**  

💡 **Hint:**  
Drop features that are redundant, uninformative, or leak the target.  
Check for near-constant values or high missingness.  
Use `df.isna()`, `df.nunique()`, and correlation checks.

✏️ *Your answer here...*

---

#### 🔑 Question 5:
**After preprocessing, what does your final input schema look like (i.e., how many numerical features and how many categorical features)? Are there any imbalance or sparsity issues to be aware of?**  

💡 **Hint:**  
Print input shapes and schema after preprocessing.  
Use value counts and histograms to detect class imbalance or feature sparsity.  
This helps you reason about model convergence or need for regularization.

✏️ *Your answer here...*


---

### 📆 Week 2: Model Development & Experimentation

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

### 📆 Week 3: Model Tuning

### 🔑 Question 1:

### 🔑 Question 2:

### 🔑 Question 3:

### 🔑 Question 4:

### 🔑 Question 5:

---

## ✅ Phase 3: Model Deployment

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
