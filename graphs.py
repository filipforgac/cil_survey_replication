import pandas as pd
import matplotlib.pyplot as plt

class PlotSetting():
    def __init__(self, marker, label, color, linestyle):
        self.marker = marker
        self.label = label
        self.color = color
        self.linestyle = linestyle

def get_csv_path(csv: str, path: list[str]) -> str:
    return f"./{"/".join(path)}/{csv}"

def show(csvs: list[str], path: list[str]) -> None:
    # style of the marker, label (in legend), color of the line, style of the line
    settings = [
        PlotSetting("p", "iCaRL", "#e3d568", "-."), 
        PlotSetting("*", "Podnet", "red", "-."),
        PlotSetting("d", "MEMO", "#e6a525", "-")
    ]  
    assert len(csvs) == len(settings), "Unequal lenght of settings and csvs"
    assert len(path) > 0, "Path is empty"

    create_csv_path = lambda csv : get_csv_path(csv, path)

    plt.figure(figsize=(10, 8))
    for csv, setting in zip(
        map(create_csv_path, csvs), 
        settings,
    ):
        # Load the CSV file
        df = pd.read_csv(csv)

        # Plot the data
        plt.plot(
            df["Number of Classes"], 
            df["Accuracy"], 
            marker=setting.marker, 
            label=setting.label, 
            color=setting.color, 
            linestyle=setting.linestyle
        )

    plt.gca().set_facecolor("#f0f0f0")  # Light grey background   
    plt.grid(color="#ffffff", linestyle="-", linewidth=2)  # White grid lines

    # Add labels
    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title(f"{path[-1]}", pad=20)
    plt.xlim(0, 102)
    plt.ylim(0, 100)

    # Show the graph
    plt.show()

if __name__ == "__main__":
    # cifar100_0_5
    show(
        ["accuracy_line_graph_icarl_cifar100_0_5.csv", 
        "accuracy_line_graph_podnet_cifar100_0_5.csv", 
        "accuracy_line_graph_memo_cifar100_0_5.csv"], 
        ["graph_data", "cifar100_0_5"]
    )
    # cifar100_0_10
    show(
        ["accuracy_line_graph_icarl_cifar100_0_10.csv", 
        "accuracy_line_graph_podnet_cifar100_0_10.csv", 
        "accuracy_line_graph_memo_cifar100_0_10.csv"], 
        ["graph_data", "cifar100_0_10"]
    )
