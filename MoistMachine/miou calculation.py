import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- DATA FROM YOUR RUN ---
classes = ["Background", "Build-Flood", "Build-Dry", "Road-Flood",
           "Road-Dry", "Water", "Tree", "Vehicle", "Pool", "Grass"]
iou_scores = [51.43, 28.14, 57.54, 30.59, 59.09, 59.07, 57.63, 0.00, 0.00, 72.10]

# Methodology Data (The "Punishment" Weights you used)
weights = [1.0, 10.0, 2.0, 10.0, 2.0, 5.0, 1.0, 2.0, 1.0, 1.0]

# Set specific colors for FloodNet classes (Approximate standard palette)
# 0:Back(Black), 1:Build-F(Red), 2:Build-D(Green), 3:Road-F(Purple), 4:Road-D(Blue)
# 5:Water(Cyan), 6:Tree(Olive), 7:Vehicle(Yellow), 8:Pool(Orange), 9:Grass(DkGreen)
colors = ['#444444', '#FF0000', '#00FF00', '#800080',
          '#0000FF', '#00FFFF', '#808000', '#FFFF00', '#FFA500', '#008000']


def plot_iou():
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Create Bar Plot
    bars = plt.bar(classes, iou_scores, color=colors, edgecolor='black', alpha=0.8)

    # Add labels on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval}%",
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.title("Per-Class Intersection over Union (IoU) on Validation Set", fontsize=14, fontweight='bold')
    plt.ylabel("IoU Score (%)", fontsize=12)
    plt.xlabel("Classes", fontsize=12)
    plt.ylim(0, 80)  # Give some headroom
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("iou_chart.png", dpi=300)
    print("Saved iou_chart.png")


def plot_weights():
    plt.figure(figsize=(10, 6))

    # Highlight the "Heavy" weights
    weight_colors = ['red' if w >= 5.0 else 'skyblue' for w in weights]

    bars = plt.bar(classes, weights, color=weight_colors, edgecolor='black')

    plt.title("Class Penalty Weights (Loss Function Strategy)", fontsize=14, fontweight='bold')
    plt.ylabel("Weight Value", fontsize=12)
    plt.xlabel("Classes", fontsize=12)

    # Add text annotation
    plt.text(5, 8, "Critical Classes (Flood) \nheavily penalized (10.0)",
             bbox=dict(facecolor='yellow', alpha=0.5))

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, f"x{height}",
                 ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("weights_chart.png", dpi=300)
    print("Saved weights_chart.png")


if __name__ == "__main__":
    plot_iou()
    plot_weights()