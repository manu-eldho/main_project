import matplotlib.pyplot as plt

# Data
models = ['XGBoost', 'Random Forest', 'KNN', 'MLP', 'ANN']
accuracy = [0.7767, 0.6733, 0.6350, 0.83, 0.79]
colors = ['blue', 'green', 'red', 'purple', 'orange']  # Colors for points

# Plot dashed gray line
plt.figure(figsize=(8,5))
plt.plot(models, accuracy, color='gray', linestyle='--', linewidth=2, label='Accuracy Trend')

# Plot points with different colors
for i in range(len(models)):
    plt.scatter(models[i], accuracy[i], color=colors[i], s=100, label=models[i])

# Add title and labels
plt.title('Model Accuracy Comparison', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.ylim(0.6, 0.85)
plt.grid(True)

# Add legend
plt.legend()

# Show plot
plt.show()
