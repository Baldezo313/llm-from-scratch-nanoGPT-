import json
import matplotlib.pyplot as plt

# Charger le log
with open("loss_log.json") as f:
    log = json.load(f)

steps = log["steps"]
train_loss = log["train"]
val_loss = log["val"]

# Détecter la fréquence de validation
freq = len(steps) // len(val_loss)
val_steps = steps[::freq][:len(val_loss)]

plt.figure(figsize=(10, 6))

plt.plot(steps, train_loss, label="Train Loss", color="blue", alpha=0.6)
plt.plot(val_steps, val_loss, label="Validation Loss", color="red", alpha=0.8)

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
