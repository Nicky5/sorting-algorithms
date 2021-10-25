import math

import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mp
import numpy as np
import random
import time

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
    bar_rects = ax.bar(range(len(array)), array, align="edge", color=color_map(data_normalizer(range(len(array)))))
    ax.set_xlim(0, len(array))
    ax.set_ylim(0, int(1.1 * len(array)))
    ax.set_title("ALGORITHM : " + algoName + "\n" + "DATA SET : " + datasetName, fontdict={'fontsize': 13, 'fontweight':'medium', 'color': '#E4365D'})
    text = ax.text(0.01, 0.95, "", transform=ax.transAxes, color="#E4365D")
    iteration = [0]

    def animate(A, rects, iteration):
        for rect, val in zip(rects, A):
            rect.set_height(val)
        iteration[0] += 1
        text.set_text("iterations : {}".format(iteration[0]))

    _ = FuncAnimation(fig, func=animate, fargs=(bar_rects, iteration), frames=generator, interval=interval, repeat=False)
    plt.show()

def faulpelz_sort(array):
    """
    Faulpelz sort takes inspiration from the developer itself and the world he lives in.
    Its time complexity i the first of a kind being only O(1).
    The concept is a ver innovative algorythm developed around elementary school named 'orbeit af die ondren schiabm'.
    It is always a good time to use the faulpelz sorting algorythm as it gives a satifing feeling of having avoided your problems.
    The only issue is it relies on the external website to be online and reliable. Just like the original algorythm
    :param array: input array
    :return: magic (∩｀-´)⊃━☆ﾟ.*･｡ﾟ
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

def menga_sort(arr):
    """
    MengaSort is inspired by "Menga" being comically small, and therefore searches for the smallest Object in an array
    to move to the current spot, while iterating through an array.
    Its time complexity equals O(n**2)
    """
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            if arr[j] < arr[i]:
                arr[i], arr[j] = arr[j], arr[i]
            yield arr

def quick_sort(array, low=None, high=None):
    """
    QuickSort is quick.
    Its average time complexity equals O(n*log(n))
    """
    if low is None: low = 0
    if high is None: high = len(array) - 1
    if low >= high:
        return
    x = array[low]
    j = low
    for i in range(low + 1, high + 1):
        if array[i] <= x:
            j += 1
            array[j], array[i] = array[i], array[j]
        yield array
    array[low], array[j] = array[j], array[low]
    yield from quick_sort(array, low, j - 1)
    yield from quick_sort(array, j + 1, high)

def bubble_sort(arr):
    """
    BubbleSort hehe funny bubbles.
    Its time complexity equals O(n**2)
    """
    n = len(arr)
    for i in range(n-1):
        for j in range(n-i-1):
            if(arr[j] > arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
            yield arr

def insertion_sort(arr):
    """
    inserts he right item in the right spot at the right time. How organized :`)
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
            yield arr
        arr[j + 1] = key

def heap_sort(arr):
    """
    Holy Eraser Apprension Portal
    """
    def heapify(arr, n, i):
        largest = i  # Initialize largest as root
        l = 2 * i + 1  # left = 2*i + 1
        r = 2 * i + 2  # right = 2*i + 2
        # See if left child of root exists and is
        # greater than root
        if l < n and arr[i] < arr[l]:
            largest = l
        # See if right child of root exists and is
        # greater than root
        if r < n and arr[largest] < arr[r]:
            largest = r
        # Change root, if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]  # swap
            # Heapify the root.
            heapify(arr, n, largest)
    n = len(arr)
    # Build a maxheap.
    # Since last parent will be at ((n//2)-1) we can start at that location.
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
        yield arr
    # One by one extract elements
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)
        yield arr

def selection_sort(arr):
    """
    Picks the right opportunities at the right time.
    """
    for i in range(len(arr)):
        # Find the minimum element in remaining
        # unsorted array
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
            yield arr
        # Swap the found minimum element with
        # the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        yield arr

def merge_sort(arr, l=None, r=None):
    """
    cool spike patterns
    """
    def merge(arr, l, m, r):
        n1 = m - l + 1
        n2 = r - m
        # create temp arrays
        L = [0] * (n1)
        R = [0] * (n2)
        # Copy data to temp arrays L[] and R[]
        for i in range(0, n1):
            L[i] = arr[l + i]
            yield arr
        for j in range(0, n2):
            R[j] = arr[m + 1 + j]
            yield arr
        # Merge the temp arrays back into arr[l..r]
        i = 0  # Initial index of first subarray
        j = 0  # Initial index of second subarray
        k = l  # Initial index of merged subarray
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            yield arr
        # Copy the remaining elements of L[], if there
        # are any
        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
            yield arr
        # Copy the remaining elements of R[], if there
        # are any
        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1
            yield arr
    # l is for left index and r is right index of the
    # sub-array of arr to be sorted
    if l is None: l = 0
    if r is None: r = len(arr)-1
    if l < r:
        # Same as (l+r)//2, but avoids overflow for
        # large l and h
        m = l + (r - l) // 2
        # Sort first and second halves
        yield from merge_sort(arr, l, m)
        yield from merge_sort(arr, m + 1, r)
        yield from merge(arr, l, m, r)

def cocktail_sort(array):
    """
    cocktail sort is a sorting algorythm based on bubblesort. The diffrence is that it's twoway and gradually gathers all unsorted items to the middle.
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

def shell_sort(arr):
    """
    flattens the curve
    """
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

def comb_sort(arr):
    """
    no honeycombs sadly
    """
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
    """
    EXTREMLY satifying. The original at least. This one is just buggy.
    """
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

def gnome_sort(array):
    """
    based of a REAL gnome sorting plants. (omg computer scientist gnome ∑(ﾟﾛﾟ〃) )
    :param array:
    :return:
    """
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

print("\ndisplay [SortingAlgorithm]\ntime [SortingAlgorithm]\n-i [show iterations count, off by default]\n"
      "-s [speed, 0 by default]]\n-d [dataset, random by default]]\n-l [dataset length, 50 by default]\n"
      "-r [dataset range, 50 by default]\n")

run = True
while run:
    commandstr = input()
    command = commandstr.split()

    drange = 50
    dlength = 50
    dinterval = 50
    dname = "random"
    for i in range(len(command)):
        if command[i] == "-d" and command[i + 1].lower() == "random":
            pass  # random is default
        elif command[i] == "-l":
            dlength = int(command[i + 1])
        elif command[i] == "-r":
            drange = int(command[i + 1])

    array = np.array(random.sample(range(0, drange), dlength))

    # please emil.
    # there has to be a better way
    if command[0].lower() == "display":
        if command[1].lower() == "faulpelzsort" or command[1].lower() == "faulpelz_sort":
            showGraph(generator=faulpelz_sort, array=array, algoName='Faulpelz Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "mengasort" or command[1].lower() == "menga_sort":
            showGraph(generator=menga_sort, array=array, algoName='Menga Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "bubblesort" or command[1].lower() == "bubble_sort":
            showGraph(generator=bubble_sort, array=array, algoName='Bubble Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "quicksort" or command[1].lower() == "quick_sort":
            showGraph(generator=quick_sort, array=array, algoName='Quick Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "insertionsort" or command[1].lower() == "insertion_sort":
            showGraph(generator=insertion_sort, array=array, algoName='Insertion Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "heapsort" or command[1].lower() == "heap_sort":
            showGraph(generator=heap_sort, array=array, algoName='Heap Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "selectionsort" or command[1].lower() == "selection_sort":
            showGraph(generator=selection_sort, array=array, algoName='Selection Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "mergesort" or command[1].lower() == "merge_sort":
            showGraph(generator=merge_sort, array=array, algoName='Merge Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "cocktailsort" or command[1].lower() == "cocktail_sort":
            showGraph(generator=cocktail_sort, array=array, algoName='Cocktail Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "shellsort" or command[1].lower() == "shell_sort":
            showGraph(generator=shell_sort, array=array, algoName='Shell Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "combsort" or command[1].lower() == "comb_sort":
            showGraph(generator=comb_sort, array=array, algoName='Comb Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "radixsort" or command[1].lower() == "radix_sort":
            showGraph(generator=radix_sort, array=array, algoName='Radix Sort', datasetName=dname, interval=dinterval)
        elif command[1].lower() == "gnomesort" or command[1].lower() == "gnome_sort":
            showGraph(generator=gnome_sort, array=array, algoName='Gnome Sort', datasetName=dname, interval=dinterval)
    # *adds some guanciale, EGGS*
    # spaghetti code alla carbonara

    # can't wait fo switch statemnt in python 3.10
    elif command[0] == "time":
        start = time.time()
        if command[1].lower() == "faulpelzsort" or command[1].lower() == "faulpelz_sort":
            faulpelz_sort(array)
        elif command[1].lower() == "mengasort" or command[1].lower() == "menga_sort":
            menga_sort(array)
        elif command[1].lower() == "bubblesort" or command[1].lower() == "bubble_sort":
            bubble_sort(array)
        elif command[1].lower() == "quicksort" or command[1].lower() == "quick_sort":
            quick_sort(array)
        elif command[1].lower() == "insertionsort" or command[1].lower() == "insertion_sort":
            insertion_sort(array)
        elif command[1].lower() == "heapsort" or command[1].lower() == "heap_sort":
            heap_sort(array)
        elif command[1].lower() == "selectionsort" or command[1].lower() == "selection_sort":
            selection_sort(array)
        elif command[1].lower() == "mergesort" or command[1].lower() == "merge_sort":
            merge_sort(array)
        elif command[1].lower() == "cocktailsort" or command[1].lower() == "merge_sort":
            cocktail_sort(array)
        elif command[1].lower() == "shellsort" or command[1].lower() == "shell_sort":
            shell_sort(array)
        elif command[1].lower() == "combsort" or command[1].lower() == "comb_sort":
            comb_sort(array)
        elif command[1].lower() == "radixsort" or command[1].lower() == "radix_sort":
            radix_sort(array)
        elif command[1].lower() == "gnomesort" or command[1].lower() == "gnome_sort":
            gnome_sort(array)
        end = time.time()
        print("Time: ", end - start)
