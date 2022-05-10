from keras import backend as K
from random import randrange
from matplotlib.patches import Circle
from tensorflow import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import copy


def get_layer_output(model, input, layer):
    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer(index=layer).output)
    intermediate_output = intermediate_layer_model(input)
    return intermediate_output

def get_layer_input(model, input, layer):
    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.get_layer(index=layer).input)
    intermediate_output = intermediate_layer_model(input)
    return intermediate_output

def get_model_graph(model, input):
    model_graph = []
    for layer_index in range(len(model.layers)):
        layer_output = get_layer_output(model, input, layer_index)
        model_graph.append(layer_output)
    return model_graph

def get_neuron_coverage(general_model_graph, current_model_graph, factor=0.5):
    above_zero = 0
    overall = 0
    for layer_index in range(len(general_model_graph)):
        for neuron_index in range(len(general_model_graph[layer_index])):
            if general_model_graph[layer_index][neuron_index] > factor or current_model_graph[layer_index][neuron_index] > factor:
                above_zero += 1
                general_model_graph[layer_index][neuron_index] = 1
            overall += 1

    return general_model_graph, above_zero/overall

def SS_coverage(general_model_graph, before_mutation_graph, after_mutation_graph, factor=0.5):
    covered = 0
    overall = 0
    for layer_index in range(len(general_model_graph)):
        for neuron_index in range(len(general_model_graph[layer_index])):
            if general_model_graph[layer_index][neuron_index] == 1:
                covered += 1
            elif (before_mutation_graph[layer_index][neuron_index] > factor > after_mutation_graph[layer_index][neuron_index]) or \
                (before_mutation_graph[layer_index][neuron_index] < factor < after_mutation_graph[layer_index][neuron_index]):
                general_model_graph[layer_index][neuron_index] = 1
                covered += 1
            overall += 1

    return general_model_graph, covered/overall

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def grad_sigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def display_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()

def adversarial_oracle(image, adversarial, dataset, factor=0.25):
    if dataset == "cifar10":
        vector = np.reshape(np.abs(adversarial - image), 3072)
    elif dataset == "mnist":
        vector = np.reshape(np.abs(adversarial - image), 784)
    elif dataset == "cad":
        vector = np.reshape(np.abs(adversarial - image), 150528)
        factor = 0.5

    distance = np.linalg.norm(vector, np.inf)
    if factor > distance > 0.02:
        return True
    else:
        return False

def target_oracle(layer_bfmt, layer_afmt, target_neuron, factor=0.5):
    msg = ""
    for index in range(len(layer_bfmt)):
        if index == target_neuron:
            continue
        if np.abs(layer_bfmt[index] - layer_afmt[index]) > factor:
            msg = "Other neurons affected"

    if (layer_bfmt[target_neuron] > 0.5 > layer_afmt[target_neuron]) or \
            (layer_bfmt[target_neuron] < 0.5 < layer_afmt[target_neuron]):
        print(msg)
        return True
    else:
        return False

def get_coverage_graph(model_graph):
    graph = [[], [], [], []]
    for x in range(len(model_graph)):
        layer = np.array(model_graph[x])
        for neuron in layer[0]:
            graph[x].append(neuron)
        graph[x] = np.array(graph[x])

    graph = np.array(graph, dtype=object)
    return graph