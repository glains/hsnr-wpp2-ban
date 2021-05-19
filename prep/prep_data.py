from pathlib import Path

import numpy as np
import pydicom


class MRT:

    def __init__(self, path):
        self.path = path
        self.layers = []


class Layer:
    def __init__(self, mrt_path, contours_path):
        self.mrt_path = mrt_path
        self.contours = contours_path

    def process(self):
        self._convert_data()

    def _convert_data(self):
        ds = pydicom.dcmread(self.mrt_path)
        window_center = ds[0x0028, 0x1050]
        window_width = ds[0x0028, 0x1051]
        target = np.zeros((len(ds.pixel_array), len(ds.pixel_array[0]), 1), dtype=np.uint8)
        for y, row in enumerate(ds.pixel_array):
            for x, pixel in enumerate(row):
                if pixel > 255:
                    target[x, y] = gray_scale_transform(pixel, 0, 255, window_center.value, window_width.value)
        return target


def prepare_data(dicom_path, output_path):
    create_structure(output_path)

    print('prepare data')
    mrts = read_data(dicom_path)
    for mrt in mrts:
        for layer in mrt.layers:
            layer.process()


def create_structure(output_path):
    print('creating project structure')
    image_path = output_path.joinpath(Path("images"))
    label_path = output_path.joinpath(Path("labels"))
    color_path = output_path.joinpath(Path("colors"))

    output_path.mkdir(exist_ok=True)
    image_path.mkdir(exist_ok=True)
    label_path.mkdir(exist_ok=True)
    color_path.mkdir(exist_ok=True)


# from pydicom import dcmread
# from pydicom.data import get_testdata_file


filename = "data/OrthosisMRT/E_BL_TSE_DIXON_CONTROL_LEG_45MIN_OPP_0043/NUTRIHEP_BASELINE_E.MR.NUTRIHEP_23NA_1H.0043.0001.2014.04.05.11.58.45.500000.11978408.IMA"


def gray_scale_transform(x, y_min, y_max, c, w):
    if x <= c - 0.5 - (w - 1) / 2:
        return y_min
    elif x > c - 0.5 + (w - 1) / 2:
        return y_max
    else:
        return ((x - (c - 0.5)) / (w - 1) + 0.5) * (y_max - y_min) + y_min


def read_data(base_path):
    mrts = []
    for mrt_dir in Path.iterdir(base_path):
        mrt = MRT(mrt_dir)

        for ima_path in mrt_dir.glob('*.IMA'):
            name = ima_path.stem + '.txt'
            contour_path = mrt_dir.joinpath('Save/Autosave').joinpath(name)
            layer = Layer(ima_path, contour_path)
            mrt.layers.append(layer)

        mrts.append(mrt)

    return mrts
