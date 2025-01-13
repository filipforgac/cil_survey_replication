import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_npy_path(npy: str, path: list[str]) -> str:
    return f"./{"/".join(path)}/{npy}"

def show(pred_npy: str, target_npy: str, path: list[str]) -> None:
    # Load prediction and target
    predictions = np.load(get_npy_path(pred_npy, path))
    true_labels = np.load(get_npy_path(target_npy, path))
    assert predictions.shape == true_labels.shape, "Prediction and target don't have the same shape"

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Visualize confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, fmt="d", cmap="BuPu", square=True)
    plt.title(f"Confusion Matrix {path[-1]}", pad=20)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Set axes ticks to jump by 20
    num_classes = cm.shape[0]
    tick_positions = np.arange(0, num_classes, 20)  # Generate ticks at intervals of 20
    plt.xticks(tick_positions, tick_positions)
    plt.yticks(tick_positions, tick_positions)

    # Add red dashed lines at 80, 80
    plt.axhline(y=80, color="red", linestyle="--", linewidth=2)  # Horizontal line
    plt.axvline(x=80, color="red", linestyle="--", linewidth=2)  # Vertical line

    plt.show()

if __name__ == "__main__":
    show("pred.npy", "target.npy", ["conf_matrices_data", "cifar100_icarl_0_20"])
    show("pred.npy", "target.npy", ["conf_matrices_data", "cifar100_podnet_0_20"])
    show("pred.npy", "target.npy", ["conf_matrices_data", "cifar100_memo_0_20"])