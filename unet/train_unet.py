from vendor import unet
import tensorflow as tf
from pathlib import Path
from matplotlib import pyplot as plt
import math

def load_data(xPath:Path, yPath:Path):
    x_files = tf.data.Dataset.list_files(str(xPath.joinpath(Path("*.png"))), shuffle=False)
    y_files = tf.data.Dataset.list_files(str(yPath.joinpath(Path("*.png"))), shuffle=False)

    filenames = tf.data.Dataset.zip((x_files, y_files))
    
    imgCount = 0

    x_vals = []
    y_vals = []

    for thing in filenames:
        x_enc = tf.io.read_file(thing[0])
        x_dec = tf.image.decode_png(x_enc, channels=1)
        
        x_vals.append(x_dec)

        y_enc = tf.io.read_file(thing[1])
        y_dec = tf.image.decode_png(y_enc, channels=1)

        y_vals.append(y_dec)

        imgCount = imgCount +1 
    
    x_data = tf.data.Dataset.from_tensor_slices(x_vals)
    y_data = tf.data.Dataset.from_tensor_slices(y_vals)

    full_data = tf.data.Dataset.zip((x_data, y_data))

    #Shuffling & Batching
    full_data = full_data.shuffle(3,reshuffle_each_iteration=False)
    full_data = full_data.batch(5)
    trainData = full_data.take(math.floor(imgCount*0.8))
    testData = full_data.skip(math.floor(imgCount*0.8)).take(math.floor(imgCount*0.1))
    validationData = full_data.skip(math.floor(imgCount*0.9)).take(math.floor(imgCount*0.1))

    return trainData, testData, validationData
if __name__=="__main__":
    train, test, val = load_data(Path("C:/Users/Colin/Documents/WPP/out/images/B_BL_TSE_DIXON_CONTROL_LEG_45MIN_OPP_0041"), Path("C:\\Users\\Colin\\Documents\\WPP\\out\\labels\\B_BL_TSE_DIXON_CONTROL_LEG_45MIN_OPP_0041"))

    for e in train.take(1):
        print('image shape=%s' % e[0].get_shape().as_list())
        print('label shape=%s' % e[1].get_shape().as_list())
    


    print("Hello world!")


