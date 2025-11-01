import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Title
st.title("🎓 Student Marks Predictor")

# Input: Study hours
hours = st.number_input("Enter number of study hours:", min_value=0.0, step=0.5)

# Sample training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([35, 52, 58, 71, 77])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
predicted_marks = model.predict([[hours]])
predictions = model.predict(X)

# Calculate error
error = mean_absolute_error(y, predictions)
st.write(f"📉 Average Prediction Error: {error:.2f} marks")

# Create chart
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Actual Marks')
ax.plot(X, model.predict(X), color='red', label='Predicted Line')
ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks")
ax.legend()
st.pyplot(fig)
plt.close(fig)  # close the figure to free memory
# Show result
if hours > 0:
    st.success(f"Predicted Marks: {predicted_marks[0]:.2f}")
else:
    st.info("👈 Enter hours to predict marks.")
