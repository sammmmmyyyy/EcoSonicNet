import matplotlib.pyplot as plt

epochs = list(range(1, 21))
accuracy = [0.0088, 0.12, 0.24, 0.35, 0.42, 0.48, 0.53, 0.58, 0.62, 0.65,
            0.68, 0.70, 0.72, 0.73, 0.74, 0.745, 0.75, 0.755, 0.76, 0.76]

plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracy, marker='o', color='teal', linewidth=2)
plt.title("Training Accuracy over Epochs", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("training_accuracy_curve.png")
plt.show()
