from random import randrange
from matplotlib.patches import Circle
from tensorflow import keras
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
import cv2



# add noise to an image
def noise_mutation(input):
    sigma = 0.05
    noise = np.random.normal(size=input.shape, scale=sigma)
    return (input + noise*255)/255


# add red, green or blue filter
def filter_mutation(input, dataset, sigma=0.3):
    if dataset == "cifar10":
        rgbl = np.array([255, 0, 0])
        random.shuffle(rgbl)
    elif dataset == "mnist":
        rgbl = np.array([255])
    elif dataset == "cad":
        rgbl = np.array([255, 0, 0])
        random.shuffle(rgbl)

    rgbl = rgbl * sigma
    input = input + np.full(shape = input.shape, fill_value = rgbl)
    return input/255


def blind_spot_manipulation(x_axis, y_axis, input, dataset, sigma=0.3):
    x = randrange(x_axis)
    y = randrange(y_axis)

    if dataset == "cifar10":
        rgbl = np.array([255, 0, 0])
        random.shuffle(rgbl)
    elif dataset == "mnist":
        rgbl = np.array([255])
    elif dataset == "cad":
        rgbl = np.array([255, 0, 0])
        random.shuffle(rgbl)


    rgbl = rgbl * sigma

    radius = randrange(int(x_axis/5))

    arr = np.zeros(input.shape)
    rr, cc = draw.disk(center=(x, y), radius=radius, shape=arr.shape)
    arr[rr, cc] = rgbl

    input = input + arr
    return input/255


def add_flip(image):
    axis = randrange(0, 3)
    image = np.flip(image, axis)
    return image/255

def add_rotation(image):
    return np.rot90(image, 1, axes=(0, 1))/255

def add_snow(image):
    img_cv2 = np.array(cv2.cvtColor(image,cv2.COLOR_RGB2HLS), dtype = np.float64)
    brightness = 2.5
    snow_threshold=160
    img_cv2[:,:,1][img_cv2[:,:,1]<snow_threshold] = img_cv2[:,:,1][img_cv2[:,:,1]<snow_threshold]*brightness
    img_cv2[:,:,1][img_cv2[:,:,1]>255] = 255
    img_cv2 = np.array(img_cv2, dtype = np.uint8)
    return cv2.cvtColor(img_cv2,cv2.COLOR_HLS2RGB)/255

def add_brightness(image):
    if randrange(2) == 0:
        brt = 0.5
    else:
        brt = 0.0
    img_cv2 = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2HLS), dtype = np.float64)
    brightness = np.random.uniform()+brt
    img_cv2[:,:,1] = img_cv2[:,:,1]*brightness
    img_cv2[:,:,1][img_cv2[:,:,1]>255] = 255
    return cv2.cvtColor(np.array(img_cv2, dtype = np.uint8), cv2.COLOR_HLS2RGB)/255

def shadow_coordinates(imshape, nr_shadows=1):
    vertices_list = []
    for i in range(nr_shadows):
        matrix = []
        for d in range(np.random.randint(3,15)):
            matrix.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))

    vertex = np.array([matrix], dtype=np.int32)
    vertices_list.append(vertex)
    return vertices_list

def add_shadow(image, nr_shadows=1):
    img_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    zero_matrix = np.zeros_like(image)

    vertices_list= shadow_coordinates(image.shape, nr_shadows)
    for vertex in vertices_list:
        cv2.fillPoly(zero_matrix, vertex, 255)

    img_cv2[:,:,1][zero_matrix[:,:,0] == 255] = img_cv2[:,:,1][zero_matrix[:,:,0] == 255] * 0.5
    img = cv2.cvtColor(img_cv2, cv2.COLOR_HLS2RGB)
    return img/255

def lines(imshape,slant,length, dataset):
    drops=[]
    if dataset == "cifar10":
        count = 3
    elif dataset == "cad":
        count = 100
    for i in range(count):
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1] - slant)
        y= np.random.randint(0, imshape[0] - length)
        drops.append((x,y))
    return drops

def add_rain(image, dataset):
    sl=10
    slant= np.random.randint(-sl,sl)

    drop_length=20
    drop_width=2
    drop_color=(200,200,200)

    rain_drops= lines(image.shape, slant,drop_length, dataset)
    for rain_drop in rain_drops:
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)

    image= cv2.blur(image,(3, 3))
    brightness = 0.9
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness

    return cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)/255

def add_blur(image):
    return cv2.blur(image,(6, 6))/255

#
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# train_path = 'Data/train'
# valid_path = 'Data/valid'
#
# train_batches = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
#     .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=1000)
# imgs, labels = next(train_batches)
#
# # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# # x_train = x_train.astype("float32") / 255
#
# def save_image(image, path=r'/tmp/adversarial/'):
#     import cv2
#     import numpy as np
#     cv2.imwrite(path, image*255)
#
#
# for i in range(10):
#     image = imgs[i]
#
#     image = image.astype(np.float64) / np.amax(image)
#     image = 255 * image
#     image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#
#
#     save_image(image)
#
#     save_image(noise_mutation(image))
#
#     save_image(blind_spot_manipulation(200, 200, image, "cad"))
#     plt.show()
#
#     plt.imshow(filter_mutation(image, "cad"))
#     plt.show()
#
#
#     plt.imshow(add_rotation(image))
#     plt.show()
#
#     plt.imshow(add_blur(image))
#     plt.show()
#
#     plt.imshow(add_shadow(image))
#     plt.show()
#
#     plt.imshow(add_brightness(image))
#     plt.show()
#
#     plt.imshow(add_snow(image))
#     plt.show()
#
#     plt.imshow(add_rain(image, "cifar10"))
#     plt.show()
