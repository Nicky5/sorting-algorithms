import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mp
import numpy as np
import random

def showGraph(generator, array, args=None, algoName='professional sorting algorythm', datasetName='Random'):
    if args is not None:
        generator = generator(array, *args)
    else:
        generator = generator(array)
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

def faul_pelz_sort(array):
    """
    Faulpelz sort takes inspiration from the developer itself and the world he lives in.
    Its time complexity i the first of a kind being only O(1).
    The concept is a ver innovative algorythm developed around elementary school named 'orbeit af die ondren schiabm'.
    It is always a good time to use the faulpelz sorting algorythm as it gives a satifing feeling of having avoided your problems.
    The only issue is it relies on the external website to be online and reliable.
    :param array: input array
    :return: sorted array
    """
    yield array
    import requests

    data = {
        "operation": "alpha",
        "outseperator": ",",
        "seperator": ",",
        "sessionid": "m",
        "usertext": numpy.array2string(array, separator=',')
    }

    response = requests.post("https://sortmylist.com/alphabetize", data=data)
    rtext = response.text.replace('[', '').replace(']', '').replace(' ', '').replace('\n', '')
    yield np.fromstring(rtext, dtype=int, sep=',')

array = np.array(random.sample(range(0, 50), 50))
showGraph(generator=faul_pelz_sort, array=array, algoName='Faulpelz Sort', datasetName='Random')