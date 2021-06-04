"""
Convenience script for loading weights and predicting on test data
"""

import train_unet
from pathlib import Path

model = train_unet.build_unet(5, 16)

model.load_weights('last_weights.ckpt')

train, test, val = train_unet.load_data(Path("C:\\Users\\Colin\\Documents\\WPP\\out\\images"),Path("C:\\Users\\Colin\\Documents\\WPP\\out\\labels"), 5)

out = model.predict(test)