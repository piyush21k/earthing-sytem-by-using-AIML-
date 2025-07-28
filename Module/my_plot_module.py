import matplotlib.pyplot as plt

def simple_line_plot(x, y, title="Line Plot", xlabel="X-axis", ylabel="Y-axis"):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()