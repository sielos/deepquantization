# in[0], axis = 0)
# test_img = np.expand_dims(x_train[1], axis = 0)
#
# initial_model_cifar10_graph = get_model_graph(model_cifar10, img)
# test_model_cifar10_graph = get_model_graph(model_cifar10, test_img)
#
# coverage_graph = get_coverage_graph(initial_model_cifar10_graph)
# current_graph = get_coverage_graph(test_model_cifar10_graph)
#
# ggg = np.array([np.zeros(3072), np.zeros(10), np.zeros(10)], dtype=object)
#
# print(neuron_coverage(coverage_graph, current_graph))
# print(SS_coverage(ggg, coverage_graph, current_graph))
#
#
# #
# # new_input = copy.deepcopy(np.array(img))
# # layer2 = np.array(get_layer_output(model_cifar10, img, 2))[0]
# # print("layer", layer2)
# # print()
# #
# # target = copy.deepcopy(np.array(layer2))
# # target_neuron = 0
# # target_value = 0
# # if target[target_neuron] < 0.25:
# #     target[target_neuron] = 1000.0
# #     target_value = 1.0
# # else:
# #     target[target_neuron] = -1000.0
# #     target_value = 0.0
# #
# # for x in range(5000):
# #     layer0 = np.array(get_layer_output(model_cifar10, new_input, 0))[0]
# #     layer1 = np.array(get_layer_output(model_cifar10, new_input, 1))[0]
# #     layer2 = np.array(get_layer_output(model_cifar10, new_input, 2))[0]
# #
# #     weights1 = model_cifar10.get_layer(index=1).get_weights()[0]
# #     bias1 = model_cifar10.get_layer(index=1).get_weights()[1]
# #
# #     weights2 = model_cifar10.get_layer(index=2).get_weights()[0]
# #     bias2 = model_cifar10.get_layer(index=2).get_weights()[1]
# #
# #
# #
# #     # h = np.matmul(weights.T, layer0) + bias
# #     # y = sigmoid(h)
# #     #
# #     # delta = (target - layer1) * grad_sigmoid(h)
# #     # dW = np.outer(delta, layer0) * 0.1
# #
# #     h1 = np.matmul(weights1.T, layer0) + bias1
# #     y1 = sigmoid(h1)
# #
# #     h2 = np.matmul(weights2.T, y1) + bias2
# #     y2 = sigmoid(h2)
# #
# #     delta2 = (target - layer2) * grad_sigmoid(h2)
# #
# #     delta1 = np.dot(weights2.T, delta2) * grad_sigmoid(h1)
# #     dW = np.outer(delta1, layer0) * 1.0
# #
# #     for i in range(dW.shape[0]):
# #         new_input = (np.reshape(dW[i], [32, 32, 3])).astype(float) + new_input
# #         new_input = np.array(np.clip(new_input, a_min=0, a_max=1))
# #
# # # new_input = np.expand_dims(new_input, axis = 0)
# # # test_model_cifar10_graph = get_model_graph(model_cifar10, new_input)
# #
# # print("layer", layer2)
# # print("target", target)
# # print()
# #
# # print(target_oracle(target, layer2, target_neuron, target_value))
# #
# #
# # new_input = np.reshape(new_input, [32, 32, 3])
# # difference = np.abs((x_train[0] - new_input).astype(float))
# # difference = np.clip(difference, a_min=0, a_max=1)
# #
# # display_image(x_train[0])
# # display_image(new_input)
# # display_image(difference)
# #
# # print(adversarial_oracle(img, new_input))
# #
# # # mutation_count = 100
# # #
# # # from mutations import noise_mutation, filter_mutation, blind_spot_manipulation
# # #
# # # for x in range(mutation_count):
# # #     mutated_img = blind_spot_manipulation(img)
# # #     test_model_cifar10_graph = get_model_graph(model_cifar10, mutated_img)
# # #     # print('Sign-Sign Coverage', SS_coverage(initial_model_cifar10_graph, test_model_cifar10_graph), '%')
# # #     print('Neuron Coverage', neuron_coverage(initial_model_cifar10_graph), '%')
# # #
# #
# #
# # # symbolic_execution(initial_model_cifar10_graph)
# #
# # # print('Sign-Sign Coverage', SS_coverage(initial_model_cifar10_graph, test_model_cifar10_graph), '%')
# # # coverage = neuron_coverage(initial_model_cifar10_graph)
# # # print('Neuron Coverage', cove


from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np

model_dir = './models/converted_model.tflite'

model = load_model('./models/simple_DNN_for_CIFAR10.h5')
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

model.summary()

# Float16 quantization
def fp16_quantization(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()

    with open('./models/model.tflite', 'wb') as f:
        f.write(tflite_quant_model)

    return tflite_quant_model

# Int* quantization
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

def int8_quantization(keras_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    return tflite_model_quant


# model = fp16_quantization(model)
tflite_model_quant = int8_quantization(model)

# Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="../models/model.tflite")
# interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

