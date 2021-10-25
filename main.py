import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mp
import numpy as np
import random

def quicksort(array, lowest, highest):
    if lowest >= highest:
        return
    x = array[lowest]
    j = lowest
    for i in range(lowest + 1, highest + 1):
        if array[i] <= x:
            j += 1
            array[j], array[i] = array[i], array[j]
        yield array
    array[lowest], array[j] = array[j], array[lowest]
    yield array

    yield from quicksort(array, lowest, j - 1)
    yield from quicksort(array, j + 1, highest)

def showGraph(generator, array, args=None, algoName='professional sorting algorythm', datasetName='Random'):
    generator = generator(array, *args)
    mp.use('Qt5Agg')
    plt.style.use('fivethirtyeight')
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
    bar_rects = ax.bar(range(len(array)), array, align="edge",
                       color=color_map(data_normalizer(range(len(array)))))
    ax.set_xlim(0, len(array))
    ax.set_ylim(0, int(1.1 * len(array)))
    ax.set_title("ALGORITHM : " + algoName + "\n" + "DATA SET : " +
                 datasetName, fontdict={'fontsize': 13, 'fontweight':
        'medium', 'color': '#E4365D'})
    text = ax.text(0.01, 0.95, "", transform=ax.transAxes, color="#E4365D")
    iteration = [0]

    def animate(A, rects, iteration):
        for rect, val in zip(rects, A):
            rect.set_height(val)
        iteration[0] += 1
        text.set_text("iterations : {}".format(iteration[0]))

    _ = FuncAnimation(fig, func=animate, fargs=(bar_rects, iteration), frames=generator, interval=50, repeat=False)
    plt.show()

array = np.array(random.sample(range(0, 50), 50))
showGraph(generator=quicksort, array=array, args=[0, len(array) - 1], algoName='Quick Sort', datasetName='Random')
