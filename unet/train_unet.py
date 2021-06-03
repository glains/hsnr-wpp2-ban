from tensorflow.python.keras.callbacks import EarlyStopping
from vendor import unet
import tensorflow as tf
from pathlib import Path
from matplotlib import pyplot as plt
import math
import argparse
from vendor.unet import unet
import pickle
from matplotlib import pyplot
import numpy as np


def load_data(xPath:Path, yPath:Path, batch_size:int):
    x_files = tf.data.Dataset.list_files(str(xPath.joinpath(Path("*.png"))), shuffle=False)
    y_files = tf.data.Dataset.list_files(str(yPath.joinpath(Path("*.png"))), shuffle=False)

    filenames = tf.data.Dataset.zip((x_files, y_files))
    
    imgCount = 0

    x_vals = []
    y_vals = []

    for thing in filenames:
        x_enc = tf.io.read_file(thing[0])
        x_dec = tf.image.decode_png(x_enc, channels=1)
        
        x_vals.append(tf.cast(x_dec, tf.float32))

        y_enc = tf.io.read_file(thing[1])
        y_dec = tf.image.decode_png(y_enc, channels=1)

        y_vals.append(tf.cast(y_dec, tf.float32))

        imgCount += 1
    
    x_data = tf.data.Dataset.from_tensor_slices(x_vals)
    y_data = tf.data.Dataset.from_tensor_slices(y_vals)
    
    full_data = tf.data.Dataset.zip((x_data, y_data))

    #Shuffling & Batching
    full_data = full_data.shuffle(3,reshuffle_each_iteration=False)
    full_data = full_data.batch(batch_size, drop_remainder=True)

    batchCount = math.floor(imgCount/batch_size)
    trainData = full_data.take(math.floor(batchCount*0.8))
    testData = full_data.skip(math.floor(batchCount*0.8)).take(math.floor(batchCount*0.1))
    validationData = full_data.skip(math.floor(batchCount*0.9)).take(math.floor(batchCount*0.1))

    return trainData, testData, validationData

def build_unet(batch_size: int, filters_orig: int) -> tf.keras.Model:
    # Input Layer
    x = tf.keras.Input(shape=(256, 256, 1), batch_size=batch_size)
    # Hidden Layer
    tensor = unet(x, filters_orig=filters_orig, out_channels=7, final_activation='linear')

    model = tf.keras.Model(inputs=x,outputs=tensor)
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )

    return model

def plotHistory(loss, accuracy, title):
    plt.plot(np.arange(len(loss))+0.5, loss, "b-", label="Loss")
    plt.plot(np.arange(len(accuracy))+0.5, accuracy, "r-", label="Accuracy")
    plt.title(title)
    plt.ylim([0,1])
    plt.xlim([0, len(loss)])
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.grid(True)

    plt.savefig('last_'+title+'.pdf')
    plt.close()

if __name__=="__main__":
    #np.random.seed(2623)
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=Path, action='store', dest='inputPath', required=True)
    parser.add_argument('-bs', type=int, action='store', dest='batchSize', default=5)
    parser.add_argument('-flt', type=int, action='store', dest='filtersOrig', default=16)

    args = parser.parse_args()

    train, test, val = load_data(args.inputPath.joinpath(Path('images')), args.inputPath.joinpath(Path('labels')), args.batchSize)

    for e in train.take(1):
        print('image shape=%s' % e[0].get_shape().as_list())
        print('label shape=%s' % e[1].get_shape().as_list())
 
    model = build_unet(args.batchSize, args.filtersOrig)

    history = model.fit(train, epochs=100, validation_data=val, callbacks=[EarlyStopping(monitor='val_loss', patience=100, restore_best_weights = True)])
    model.save_weights('last_weights.ckpt')
    pickle.dump(history.history, open('last_history.dmp', 'wb'))#failsafe

    plotHistory(history.history["loss"], history.history["sparse_categorical_accuracy"], "training")
    plotHistory(history.history["val_loss"], history.history["val_sparse_categorical_accuracy"], "validation")





