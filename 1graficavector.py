import matplotlib.pyplot as plt
from warnings import filterwarnings


filterwarnings('ignore')

def vect_fig():
    print("creando")
    fig, ax= plt.subplots()
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_position("zero")
    for spine in ["right", "top"]:
        ax.spines[spine].set_color("none")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.grid()
    v=[6,4]
    v2= [7, 5]
    plt.figure()
    ax.annotate(".", xy=v, xytext=[0,0],arrowprops = dict(facecolor="blue", shrink=0, alpha=0.7, width=0.5))
    ax.text(1.1 * v[0], 1.1 * v[1], v)
    ax.annotate(".", xy=v2, xytext=[0,0],arrowprops = dict(facecolor="blue", shrink=0, alpha=0.7, width=0.5))
    ax.text(1.1 * v2[0], 1.1 * v2[1], v2)
    fig.savefig("test.png")

vect_fig()