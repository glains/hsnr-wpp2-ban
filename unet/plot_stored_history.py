"""
Convencience script for loading and plotting history
"""

import pickle
import train_unet

history = pickle.load(open('last_history.dmp', 'rb'))

train_unet.plotHistory(history["loss"],history["sparse_categorical_accuracy"],"training")
train_unet.plotHistory(history["val_loss"], history["val_sparse_categorical_accuracy"], "validation")