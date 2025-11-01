# -GenAI Basics-
Foundational Machine Learning and Generative AI projects built using Python and Streamlit — covering linear regression, gradient descent, and core learning concepts.

Each project focuses on understanding how models predict and learn, step by step — forming a strong base for advanced Gen AI concepts.


📁 Projects Included —

🧮 1. Marks Predictor

A simple Linear Regression app that predicts a student's marks based on the number of study hours.

🔹 Features
Input study hours → get predicted marks instantly

Built using scikit-learn LinearRegression

Streamlit-based user interface

Displays prediction error (Mean Absolute Error)

Simple, practical demo of real-world ML usage

🔹 Run Command

streamlit run marks_predictor.py

🧠 2. Gradient Descent Visualizer

An interactive app that shows how models actually learn using Gradient Descent, the same method behind neural networks and Gen AI models.

🔹 Features

Real-time visualization of the learning process

Displays current epoch, loss, slope, and intercept

The red line moves to fit blue data points — showing “learning in motion”

Built from scratch using only NumPy + Matplotlib

🔹 Run Command

streamlit run gradient_descent_visual.py


🧩 Tech Stack
Python 3.11+
Streamlit — Interactive Web UI
NumPy — Matrix Operations
Matplotlib — Visualizations
Scikit-learn — ML Library for Marks Predictor

⚙️ Setup Instructions

1. Clone this repository:
git clone https://github.com/anushkapundir1/GenAI-Learning.git

cd GenAI-Learning

2. Create and activate a virtual environment:

python -m venv genai-env

.\genai-env\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Run any project:

streamlit run filename.py

📚 Learnings from These Projects

Understanding the relationship between input (X) and output (y)

How regression models make predictions

How loss decreases during model training

The role of slope (m) and intercept (c)

The connection between Linear Regression and Neural Networks


👩‍💻 Author

Anushka Pundir

Aspiring Gen AI Engineer | Python & ML Enthusiast


“First, understand how a line learns — then you can teach machines to think.” ✨
