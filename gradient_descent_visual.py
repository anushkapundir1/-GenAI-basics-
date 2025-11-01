import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("🧠 Visualizing Gradient Descent — How Models Learn")

#sample data(hours vs marks)
X= np.array([1,2,3,4,5])
y= np.array([35,52,58,71,77])

# Hyperparameters
lr=0.01   # learning rate
epochs = 100

# initialize parameters randomly
m = np.random.randn()
c = np.random.randn()

#function to predict y
def predict(X, m, c):
    return m*X+c

# Training loop with visualization
progress = st.empty()
chart = st.empty()

for epoch in range(epochs):
    y_pred = predict(X, m, c)
    error = y - y_pred

    # Calculate gradients
    dm = -2 * np.mean(X * error)
    dc = -2 * np.mean(error)

    # Update parameters
    m -= lr * dm
    c -= lr * dc

    # Calculate loss (Mean Squared Error)
    loss = np.mean(error ** 2)

    # Display progress
    progress.text(f"Epoch: {epoch+1}/{epochs} | Loss: {loss:.4f} | m: {m:.3f}, c: {c:.3f}")

    # Plot each step
    fig, ax = plt.subplots()
    ax.scatter(X, y, color="blue", label="Actual Data")
    ax.plot(X, y_pred, color="red", label="Model Prediction")
    ax.legend()
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.set_title(f"Epoch {epoch+1} | Loss: {loss:.2f}")

    chart.pyplot(fig)
    plt.close(fig)   # 👈 close the figure after displaying
    time.sleep(0.1)

st.success(f"✅ Training Complete! Final Equation: Marks = {m:.2f} * Hours + {c:.2f}")
