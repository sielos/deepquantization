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
    # input = input.astype(int)

    return input/255


def add_flip(image):
    axis = randrange(0, 3)
    image = np.flip(image, axis)
    return image/255

def add_rotation(image):
    return np.rot90(image, 1, axes=(0, 1))/255


def add_snow(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    # Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    brightness_coefficient = 2.5
    snow_point=160
    ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient
    ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255
    ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    ## Conversion to RGB
    return image_RGB/255

# sunny and shady
def add_brightness(image):
    if randrange(2) == 0:
        brt = 0.5
    else:
        brt = 0.0
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    random_brightness_coefficient = np.random.uniform()+brt
    ## generates value between 0.5 and 1.5
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient
    ## scale pixel values up or down for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255] = 255
    ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    ## Conversion to RGB
    return image_RGB/255

# shadows
def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(3,15)):
            ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
    vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
    vertices_list.append(vertices)
    return vertices_list
    ## List of shadow vertices

def add_shadow(image,no_of_shadows=1):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    # Conversion to HLS
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows)
    #3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255)
    ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5
    ## if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)
    ## Conversion to RGB
    return image_RGB/255

# rain
def generate_random_lines(imshape,slant,drop_length, dataset):
    drops=[]
    if dataset == "cifar10":
        count = 3
    elif dataset == "cad":
        count = 100
    for i in range(count):
        # If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops

def add_rain(image, dataset):
    imshape = image.shape
    slant_extreme=10
    slant= np.random.randint(-slant_extreme,slant_extreme)
    drop_length=20
    drop_width=2
    drop_color=(200,200,200) ## a shade of gray
    rain_drops= generate_random_lines(imshape,slant,drop_length, dataset)
    for rain_drop in rain_drops:
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image,(3, 3)) ## rainy view are blurry
    brightness_coefficient = 0.9 ## rainy days are usually shady
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB/255

def add_blur(image):
    return cv2.blur(image,(6, 6))/255 ## rainy view are blurry


# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# #
# # train_path = 'Data/train'
# # valid_path = 'Data/valid'
# #
# # train_batches = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
# #     .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=1000)
# # imgs, labels = next(train_batches)
#
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train = x_train.astype("float32") / 255
#
# for i in range(10):
#     image = x_train[i]
#
#     image = image.astype(np.float64) / np.amax(image)
#     image = 255 * image
#     image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#
#
#     plt.imshow(image)
#     plt.show()
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
#     plt.imshow(add_flip(image))
#     plt.show()
#
#     plt.imshow(add_rain(image, "cifar10"))
#     plt.show()
