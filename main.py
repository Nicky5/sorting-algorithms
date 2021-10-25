# import all the modules
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mp
import numpy as np
import random

mp.use('Qt5Agg')

# quicksort function
def quicksort(a, l, r):
    if l >= r:
        return
    x = a[l]
    j = l
    for i in range(l + 1, r + 1):
        if a[i] <= x:
            j += 1
            a[j], a[i] = a[i], a[j]
        yield a
    a[l], a[j] = a[j], a[l]
    yield a

    # yield from statement used to yield
    # the array after dividing
    yield from quicksort(a, l, j - 1)
    yield from quicksort(a, j + 1, r)

# function to plot bars
def showGraph():
    # for random unique values
    n = int(input("enter array size\n"))
    a = [i for i in range(1, n + 1)]
    random.shuffle(a)
    datasetName = 'Random'
    # generator object returned by the function
    generator = quicksort(a, 0, n - 1)
    algoName = 'Quick Sort'
    # style of the chart
    plt.style.use('fivethirtyeight')
    # set colors of the bars
    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap(
        "my_map",
        {
            "red": [(0, 1.0, 1.0),
                    (1.0, .5, .5)],
            "green": [(0, 0.5, 0.5),
                      (1.0, 0, 0)],
            "blue": [(0, 0.50, 0.5),
                     (1.0, 0, 0)]
        }
    )
    fig, ax = plt.subplots()
    # bar container
    bar_rects = ax.bar(range(len(a)), a, align="edge",
                       color=color_map(data_normalizer(range(n))))
    # setting the limits of x and y axes
    ax.set_xlim(0, len(a))
    ax.set_ylim(0, int(1.1 * len(a)))
    ax.set_title("ALGORITHM : " + algoName + "\n" + "DATA SET : " +
                 datasetName, fontdict={'fontsize': 13, 'fontweight':
        'medium', 'color': '#E436  5D'})
    # the text to be shown on the upper left indicating the number of iterations
    # transform indicates the position with relevance to the axes coordinates.
    text = ax.text(0.01, 0.95, "", transform=ax.transAxes, color="#E4365D")
    iteration = [0]
    def animate(A, rects, iteration):
        for rect, val in zip(rects, A):
            # setting the size of each bar equal to the value of the elements
            rect.set_height(val)
        iteration[0] += 1
        text.set_text("iterations : {}".format(iteration[0]))
    # call animate function repeatedly
    anim = FuncAnimation(fig, func=animate,
                         fargs=(bar_rects, iteration), frames=generator, interval=50,
                         repeat=False)
    plt.show()

showGraph()