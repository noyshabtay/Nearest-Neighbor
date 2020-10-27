"""
Introduction to Machine Learning TAU Course.
Homework #1
"""

from sklearn.datasets import fetch_openml
from matplotlib import pyplot
import numpy

#------------Algorithem-----------
def k_nearest_neighbors(images, labels, query, k, n):
    dists = [] #an ArrayList to store distances for each image to the query image.
    for i in range(n):
        dist = numpy.linalg.norm(images[i] - query) #l2 distance computation.
        dists.append((dist, labels[i]))
    dists.sort() #in place sorting by distances in ascending order.
    return most_common_label(dists[:k])

def most_common_label(d_l_lst):
    labels, cnt, most_common = {}, 0, None
    for tup in d_l_lst:
        label = tup[1]
        labels[label] = labels.get(label, 0) + 1
        if labels[label] > cnt:
            cnt, most_common = labels[label], label
    return most_common

#------------Checking accuracy-----------
def prediction_accuracy(train_images, train_labels, test_images, test_labels, k, n):
    correct_labels = 0
    for i in range(len(test_labels)):
        prediction = k_nearest_neighbors(train_images, train_labels, test_images[i], k, n)
        if prediction == test_labels[i]: correct_labels += 1
    return correct_labels / len(test_labels)
#----------------------------------------
def main():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']
    constant = 10000
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train_images = data[idx[:constant], :].astype(int)
    train_labels = labels[idx[:constant]]
    test_images = data[idx[constant:], :].astype(int)
    test_labels = labels[idx[constant:]]
    n = 1000
    
    #section b
    print("------------ Section b -----------")
    k = 10
    accuracy = prediction_accuracy(train_images, train_labels, test_images, test_labels, k, n)
    print("n={} k={}: percentage of correct labels is {}".format(n, k, accuracy))

    #section c
    print("------------ Section c -----------")
    k_arr = [i for i in range(1, 101)]
    accuracy_arr = []
    for k in range(1, 101):
        print("k = {}".format(k))
        accuracy_arr.append(prediction_accuracy(train_images, train_labels, test_images, test_labels, k, n))
    pyplot.xlabel("k") 
    pyplot.ylabel("Accuracy") 
    pyplot.plot(k_arr,accuracy_arr) 
    pyplot.show() 

    #section d
    print("------------ Section d -----------")
    k = 1
    k_arr = [i for i in range(100, 5100, 100)]
    accuracy_arr = []
    for n in range(100, 5100, 100):
        print("n = {}".format(n))
        accuracy_arr.append(prediction_accuracy(train_images, train_labels, test_images, test_labels, k, n))
    pyplot.xlabel("n") 
    pyplot.ylabel("accuracy") 
    pyplot.plot(k_arr,accuracy_arr) 
    pyplot.show()

main()