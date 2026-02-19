import matplotlib.pyplot as plt

def setup_plot(title, x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return ax

def finalize_plot(ax):
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
