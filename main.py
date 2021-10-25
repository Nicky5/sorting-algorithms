import math

import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation
import matplotlib as mp
import numpy as np
import random

def showGraph(generator, array, args=None, algoName='professional sorting algorythm', datasetName='Random', interval=50):
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

    _ = FuncAnimation(fig, func=animate, fargs=(bar_rects, iteration), frames=generator, interval=interval, repeat=False)
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

def cocktailSort(array):
    """
    cocktail sort is a sorting algorythm based on bubblesort. The diffrence is that it's twoway and gradually gathers all unsorted items to the middle.
    :param array:
    :return:
    """
    n = len(array)
    swapped = True
    start = 0
    end = n - 1
    while (swapped == True):
        swapped = False

        for i in range(start, end):
            if (array[i] > array[i + 1]):
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True
                yield array
        if (swapped == False):
            break
        swapped = False
        end = end - 1
        for i in range(end - 1, start - 1, -1):
            if (array[i] > array[i + 1]):
                array[i], array[i + 1] = array[i + 1], array[i]
                swapped = True
                yield array
        start = start + 1
    yield array

def shellSort(arr):
    gap = len(arr) // 2
    while gap > 0:
        i = 0
        j = gap
        while j < len(arr):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j += 1
            k = i
            while k - gap > -1:
                if arr[k - gap] > arr[k]:
                    arr[k - gap], arr[k] = arr[k], arr[k - gap]
                    yield arr
                k -= 1
        gap //= 2

def combSort(arr):
    def getNextGap(gap):
        gap = (gap * 10) // 13
        if gap < 1:
            return 1
        return gap

    n = len(arr)
    gap = n
    swapped = True
    while gap != 1 or swapped == 1:
        gap = getNextGap(gap)
        swapped = False
        for i in range(0, n - gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                swapped = True
                yield arr

def radix_sort(array):
    RADIX = 10
    buckets = [[] for i in range(RADIX)]
    maxLength = False
    placement = 1
    while not maxLength:
        maxLength = True
        for i in array:
            tmp = i // placement
            buckets[tmp % RADIX].append(i)
            if maxLength and tmp > 0:
                maxLength = False
        a = 0
        for bucket in buckets:
            for i in bucket:
                yield array
                array[a] = i
                yield array
                a += 1
            bucket.clear()
        placement *= RADIX

def gnomeSort(array):
    index = 0
    n = len(array)
    while index < n:
        if index == 0:
            index = index + 1
        if array[index] >= array[index - 1]:
            index = index + 1
        else:
            array[index], array[index - 1] = array[index - 1], array[index]
            yield array
            index = index - 1
    yield array

array = lambda :np.array(random.sample(range(0, 64), 64))
showGraph(generator=faul_pelz_sort, array=array(), algoName='Faulpelz Sort', datasetName='Random', interval=50)
showGraph(generator=cocktailSort, array=array(), algoName='Cocktail sort', datasetName='Random', interval=10)
showGraph(generator=shellSort, array=array(), algoName='Shell sort', datasetName='Random', interval=50)
showGraph(generator=combSort, array=array(), algoName='Comb sort', datasetName='Random', interval=10)
showGraph(generator=radix_sort, array=array(), algoName='Radix sort', datasetName='Random', interval=10)
showGraph(generator=gnomeSort, array=array(), algoName='Gnome sort', datasetName='Random', interval=10)