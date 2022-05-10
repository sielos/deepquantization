import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3

import argparse
from tensorflow import keras
from tensorflow.keras import layers
import random
from skimage import draw
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import glob
import os

from lib.coverage import *
from lib.mutations import *
from test_DNN import *
from lib.quantize import *
from lib.models import *

from pathlib import Path

Path("/tmp/testing").mkdir(parents=True, exist_ok=True)
Path("/tmp/adversarial").mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--quantize", help="quantize neural network", action="store_true")
parser.add_argument("--coverage", help="coverage", choices=("nc", "ssc"))
parser.add_argument("--generator", help="fuzzing or coverage guided fuzzing", choices=("fuzzing", "cfg", "fuzzing_cfg"))
parser.add_argument("--dataset", help="dataset used", choices=("mnist", "cifar10", "cad"))
parser.add_argument("--test", help="tests existing original and adversarial images", action="store_true")

args = parser.parse_args()
cfg = False
fuzzing = False

if args.generator == "fuzzing" or args.generator == "cfg" or args.generator == "fuzzing_cfg":
    files = glob.glob('/tmp/testing/*')
    for f in files:
        os.remove(f)

    files = glob.glob('/tmp/adversarial/*')
    for f in files:
        os.remove(f)

    Path("/tmp/testing").mkdir(parents=True, exist_ok=True)
    Path("/tmp/adversarial").mkdir(parents=True, exist_ok=True)

if args.dataset == "mnist":
    model, train_images, train_labels = get_mnist_model()
    cvrg_graph = np.array([np.zeros(784), np.zeros(100), np.zeros(10)], dtype=object)

if args.dataset == "cifar10":
    model, train_images, train_labels = get_cifar10_model()
    cvrg_graph = np.array([np.zeros(3072), np.zeros(100), np.zeros(10)], dtype=object)

if args.dataset == "cad":
    model, train_images, train_labels = get_cad_model()
    cvrg_graph = np.array([np.zeros(150528), np.zeros(100), np.zeros(10)], dtype=object)


if args.generator == "fuzzing":
    fuzzing = True
elif args.generator == "cfg":
    cfg = True
elif args.generator == "fuzzing_cfg":
    fuzzing = True
    cfg = True

if args.generator == "fuzzing" or args.generator == "cfg" or args.generator == "fuzzing_cfg":
    for i in range(5):
        img_idx = randrange(train_images.shape[0])
        img = np.expand_dims(train_images[img_idx], axis=0)
        # img = train_images[img_idx]

        if args.dataset == "cifar10":
            label = np.where(train_labels[img_idx] == 1)[0][0]
        elif args.dataset == "mnist":
            label = train_labels[img_idx]
        elif args.dataset == "cad":
            label = np.where(train_labels[img_idx] == 1)[0][0]

        cvrg_graph = test_neural_network(img, model, cvrg_graph, label, i, args.coverage, fuzzing, cfg, args.dataset)



if args.quantize:
    tflite_model = get_tflite_model(model)

    def representative_data_gen():
      for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Save the unquantized model:
    tflite_model_file = tflite_models_dir / (str(args.dataset) + "_model.tflite")
    tflite_model_file.write_bytes(tflite_model)

    # Save the quantized model:
    tflite_model_quant_file = tflite_models_dir / (str(args.dataset) + "_model.tflite")
    tflite_model_quant_file.write_bytes(tflite_model_quant)


    if args.dataset == "cifar10":
        train_l = []
        for index in range(len(train_labels)):
            train_l.append(np.where(train_labels[index] == 1)[0][0])
        train_l = np.array(train_l)

        train_labels = []
        train_labels = train_l
    if args.dataset == "cad":
        train_l = []
        for index in range(len(train_labels)):
            train_l.append(np.where(train_labels[index] == 1)[0][0])
        train_l = np.array(train_l)

        train_labels = []
        train_labels = train_l


    # # Singular evaluation
    # test_model(tflite_model_file, 1, "Float", train_images, train_labels)
    # test_model(tflite_model_quant_file, 1, "Float", train_images, train_labels)
    # print()

    # entire dataset evaluation
    print("Unquantized model")
    evaluate_model(tflite_model_file, "Float", train_images, train_labels)
    print("Quantized model")
    evaluate_model(tflite_model_quant_file, "Float", train_images, train_labels)

    # adversarial example evaluation
    image_list = []
    label_list = []
    for filename in glob.glob('/tmp/adversarial/*.png'): #assuming gif
        im=Image.open(filename)
        im = np.array(im)
        image_list.append(im)
        label_list.append(int(filename[17]))

    image_list = np.array(image_list)
    label_list = np.array(label_list)


    # # Singular evaluation
    # test_model(tflite_model_file, 1, "Float", image_list, label_list)
    # test_model(tflite_model_quant_file, 1, "Float", image_list, label_list)

    # entire dataset evaluation on adversarial
    print()
    print("Unquantized model")
    evaluate_model(tflite_model_file, "Float", image_list, label_list)
    print("Quantized model")
    evaluate_model(tflite_model_quant_file, "Float", image_list, label_list)


if args.test:
    tflite_model = get_tflite_model(model)

    def representative_data_gen():
      for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    tflite_models_dir = pathlib.Path("/tmp/mnist_tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Save the unquantized model:
    tflite_model_file = tflite_models_dir / (str(args.dataset) + "_model.tflite")
    tflite_model_file.write_bytes(tflite_model)

    if args.dataset == "cifar10":
        train_l = []
        for index in range(len(train_labels)):
            train_l.append(np.where(train_labels[index] == 1)[0][0])
        train_l = np.array(train_l)

        train_labels = []
        train_labels = train_l
    if args.dataset == "cad":
        train_l = []
        for index in range(len(train_labels)):
            train_l.append(np.where(train_labels[index] == 1)[0][0])
        train_l = np.array(train_l)

        train_labels = []
        train_labels = train_l


    # adversarial example evaluation
    image_list = []
    label_list = []
    for filename in glob.glob('/tmp/true_adversarial/*.png'): #assuming png
        im=Image.open(filename)
        im = np.array(im)
        image_list.append(im)
        label_list.append(int(filename[22]))

    image_list = np.array(image_list)
    label_list = np.array(label_list)

    ori_image_list = []
    ori_label_list = []
    for filename in glob.glob('/tmp/original/*.png'): #assuming png
        im=Image.open(filename)
        im = np.array(im)
        ori_image_list.append(im)
        ori_label_list.append(int(filename[22]))

    ori_image_list = np.array(ori_image_list)
    ori_label_list = np.array(ori_label_list)


    # Singular evaluation
    predictions = run_tflite_model(tflite_model_file, [0], ori_image_list, ori_label_list)
    plt.imshow(ori_image_list[0])
    template = "Float" + " Model \n True:{true}, Predicted:{predict}"
    _ = plt.title(template.format(true=str(ori_label_list[0]), predict=str(predictions[0])))
    plt.grid(False)
    plt.show()


    for i in range(image_list.shape[0]):
        predictions = run_tflite_model(tflite_model_file, [i], image_list, label_list)
        if str(predictions[0]) != str(label_list[i]):
            test_model(tflite_model_file, i, "Float", image_list, label_list)
