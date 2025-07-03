# 🎓 Prediction of Student Performance Using Deep Neural Networks

A deep learning project that predicts student academic outcomes using Learning Management System (LMS) data. This project applies educational data mining techniques to help identify students at risk and improve learning strategies.

## 📌 Overview

This model analyzes anonymized LMS interaction data—such as quiz scores, logins, and assignment submissions—to predict student performance. Built using Python and Scikit-learn, the neural network achieved around **90% accuracy**. Dropout regularization and hyperparameter tuning improved generalization and reduced overfitting by ~20%.

## 🔧 Technologies Used

- Python
- Scikit-learn
- NumPy & Pandas
- Matplotlib & Seaborn
- Deep Neural Networks

## 📈 Key Features

- Preprocessing and cleaning of LMS datasets.
- Trained deep neural network with dropout and tuning.
- Visual evaluation using confusion matrix, accuracy/loss plots.
- Insights to support educational interventions.

## 📊 Results

- ✅ ~90% prediction accuracy
- ✅ Overfitting reduced by ~20% with dropout
- ✅ Strong generalization across test data

## 📁 Dataset

The model was trained on anonymized educational data. Features include:
- Quiz Scores  
- Login Frequency  
- Assignment Submissions  
- Participation Activity

## 📉 Visualizations

- Confusion Matrix
- Accuracy vs Epochs
- Loss vs Epochs

## 🧠 Future Work

- Integration with real-time LMS dashboards
- Adding more behavioral features
- Deployment as a web-based tool

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/student-performance-dnn.git

# Navigate to project directory
cd student-performance-dnn

# Install dependencies
pip install -r requirements.txt

# Run the Project
python Main.py

# Or click run.bat file without above command to Run the Project