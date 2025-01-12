from matplotlib import pyplot as plt
import numpy as np

def get_npy_path(npy: str, path: list[str]) -> str:
    return f"./{"/".join(path)}/{npy}"

def show(npy: str, path: list[str]) -> None:
    assert len(path) > 0, "Path is empty"
    plt.title(f"cf_{path[-1]}", pad=20)
    img_array = np.load(get_npy_path(npy, path))
    img_array = img_array.reshape((100, 100))
    plt.imshow(img_array, cmap="binary")
    plt.show()

if __name__ == "__main__":
    show("pred.npy", ["logs", "icarl_cifar100_0_5"])