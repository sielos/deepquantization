import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from skimage import draw
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import cv2

from lib.coverage import *
from lib.mutations import *

def save_image(image, path):
    import cv2
    import numpy as np
    cv2.imwrite(path, image*255)


def test_neural_network(image, model_cifar10, coverage_graph, image_count, m, coverage, fuzzing, cfg, dataset):
    save_image(image[0], r'/tmp/testing/original' + str(image_count) + str(m) + '.png')

    model_graph = get_model_graph(model_cifar10, image)
    image = image[0]/255

    neuron_coverage = 0
    x_axis = image.shape[0]
    y_axis = image.shape[1]

    if fuzzing:
        print("Starting Fuzzing Mutations")
        cvrg = [-2, -1, 0]
        count = 0

        while (cvrg[0] != cvrg[1] != cvrg[2]) or count < 100:
            image = image.astype(np.float64) / np.amax(image)
            image = 255 * image
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            if dataset == "mnist" or dataset == "cifar10":
                mutation = randrange(3)
            elif dataset == "cad":
                mutation = randrange(10)


            mutated_image = []

            if mutation == 0:
                mutated_image = copy.deepcopy(blind_spot_manipulation(x_axis, y_axis, image, dataset))
            if mutation == 1:
                mutated_image = copy.deepcopy(noise_mutation(image))
            if mutation == 2:
                mutated_image = copy.deepcopy(filter_mutation(image, dataset))

            if mutation == 3:
                mutated_image = copy.deepcopy(add_flip(image))
            if mutation == 4:
                mutated_image = copy.deepcopy(add_rotation(image))
            if mutation == 5:
                mutated_image = copy.deepcopy(add_snow(image))

            if mutation == 6:
                mutated_image = copy.deepcopy(add_brightness(image))
            if mutation == 7:
                mutated_image = copy.deepcopy(add_shadow(image))
            if mutation == 8:
                mutated_image = copy.deepcopy(add_rain(copy.deepcopy(image), dataset))

            if mutation == 9:
                mutated_image = copy.deepcopy(add_blur(image))

            mutated_image = np.expand_dims(mutated_image, axis=0)
            image = image/255
            mutated_model_graph = get_model_graph(model_cifar10, mutated_image)

            if coverage == "nc":
                coverage_graph, neuron_coverage = get_neuron_coverage(coverage_graph, get_coverage_graph(mutated_model_graph))
            elif coverage == "ssc":
                coverage_graph, neuron_coverage = SS_coverage(coverage_graph, get_coverage_graph(model_graph), get_coverage_graph(mutated_model_graph))
            count += 1

            if count%10 == 0:
                cvrg[0] = cvrg[1]
                cvrg[1] = cvrg[2]
                cvrg[2] = neuron_coverage
                print(neuron_coverage)

            if adversarial_oracle(image, mutated_image, dataset):
                print("Adversarial Image Found")
            save_image(mutated_image[0], r'/tmp/adversarial/' + str(image_count) + '_' + str(count) + str(m) + 'adversarial_fuzz.png')
            save_image(mutated_image[0], r'/tmp/testing/' + str(image_count) + '_' + str(count) + str(m) + 'adversarial_fuzz.png')
            if np.amax(mutated_image) >= np.amax(image):
                save_image(mutated_image[0] - image, r'/tmp/testing/' + str(image_count) + '_' + str(count) + str(m) + 'difference_fuzz.png')
            else:
                save_image(image - mutated_image, r'/tmp/testing/' + str(image_count) + '_' + str(count) + str(m) + 'difference_fuzz.png')
        image = np.expand_dims(image, axis=0)

    if cfg:
        print("Starting Coverage Guided Mutations")

        target_layer = randrange(1, 3)
        layer = np.array(get_layer_output(model_cifar10, image, target_layer))[0]
        neuron_index = int(get_random_neuron(coverage_graph[target_layer]))

        target = copy.deepcopy(np.array(layer))
        new_input = copy.deepcopy(np.array(image))

        if target[neuron_index] < 0.5:
            target[neuron_index] = 10.0
        else:
            target[neuron_index] = -10.0

        for x in range(1000):
            layer0 = np.array(get_layer_output(model_cifar10, new_input, 0))[0]
            layer1 = np.array(get_layer_output(model_cifar10, new_input, 1))[0]
            layer2 = np.array(get_layer_output(model_cifar10, new_input, 2))[0]

            weights1 = model_cifar10.get_layer(index=1).get_weights()[0]
            bias1 = model_cifar10.get_layer(index=1).get_weights()[1]

            weights2 = model_cifar10.get_layer(index=2).get_weights()[0]
            bias2 = model_cifar10.get_layer(index=2).get_weights()[1]

            dW = []

            if target_layer == 1:
                h = np.matmul(weights1.T, layer0) + bias1
                y = sigmoid(h)

                delta = (target - layer1) * grad_sigmoid(h)
                dW = np.outer(delta, layer0) * 0.1

            elif target_layer == 2:
                h1 = np.matmul(weights1.T, layer0) + bias1
                y1 = sigmoid(h1)

                h2 = np.matmul(weights2.T, y1) + bias2
                y2 = sigmoid(h2)

                delta2 = (target - layer2) * grad_sigmoid(h2)

                delta1 = np.dot(weights2, delta2) * grad_sigmoid(h1)
                dW = np.outer(delta1, layer0) * 1.0

            for i in range(dW.shape[0]):
                if dataset == "cifar10":
                    new_input = (np.reshape(dW[i], [32, 32, 3])).astype(float) + new_input
                elif dataset == "mnist":
                    new_input = (np.reshape(dW[i], [28, 28])).astype(float) + new_input

                new_input = np.array(np.clip(new_input, a_min=0, a_max=1))

            if target_oracle(layer, np.array(get_layer_output(model_cifar10, new_input, target_layer))[0], neuron_index, target[neuron_index]):
                print("Passed Target Oracle")
                break

        if adversarial_oracle(image, new_input, dataset):
            print("Passed Test Oracle")
            print("Adversarial Image Created")
            save_image(new_input[0], r'/tmp/adversarial/' + str(image_count) + str(m) + 'adversarial_cfg.png')
            save_image(new_input[0], r'/tmp/testing/' + str(image_count) + str(m) + 'adversarial_cfg.png')
            save_image(new_input[0] - image[0], r'/tmp/testing/' + str(image_count) + str(m) + 'difference_cfg.png')


        mutated_model_graph = get_model_graph(model_cifar10, new_input)
        if coverage == "nc":
            coverage_graph, neuron_coverage = get_neuron_coverage(coverage_graph, get_coverage_graph(mutated_model_graph))
        elif coverage == "ssc":
            coverage_graph, neuron_coverage = SS_coverage(coverage_graph, get_coverage_graph(model_graph), get_coverage_graph(mutated_model_graph))

        print(neuron_coverage)
        print()

    return coverage_graph


def get_uncovered_neuron_index(layer):
    for neuron_index in range(len(layer)):
        if layer[neuron_index] == 0:
            return neuron_index
    return None

def get_random_neuron(layer):
    neuron_index_array = []
    for neuron_index in range(len(layer)):
        if layer[neuron_index] == 0:
            neuron_index_array.append(neuron_index)

    neuron_index_array = np.array(neuron_index_array)
    if len(neuron_index_array) == 0:
        neuron_index_array = np.append(neuron_index_array, int(0))

    return random.choice(neuron_index_array)
