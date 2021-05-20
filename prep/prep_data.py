from pathlib import Path

import numpy as np
import pydicom

import cv2

import prep.contourreader as contourreader


def gray_scale_transform(x, y_min, y_max, c, w):
    if x <= c - 0.5 - (w - 1) / 2:
        return y_min
    elif x > c - 0.5 + (w - 1) / 2:
        return y_max
    else:
        return ((x - (c - 0.5)) / (w - 1) + 0.5) * (y_max - y_min) + y_min


class MRT:

    def __init__(self, path):
        self.path = path
        self.layers = []


class Layer:
    def __init__(self, mrt_path, contours_path,contour_reader):
        self.mrt_path = mrt_path
        self.contours = contours_path
        self.gray_img = None
        self.contour_reader = contour_reader

    def process(self):
        self._convert_data()
        contour = self._remove_background()
        manual_contour = self.contour_reader.parseFile(self.contours)

    def _convert_data(self):
        ds = pydicom.dcmread(self.mrt_path)
        window_center = ds[0x0028, 0x1050]
        window_width = ds[0x0028, 0x1051]
        target = np.zeros((len(ds.pixel_array), len(ds.pixel_array[0]), 1), dtype=np.uint8)
        for y, row in enumerate(ds.pixel_array):
            for x, pixel in enumerate(row):
                if pixel > 255:
                    target[x, y] = gray_scale_transform(pixel, 0, 255, window_center.value, window_width.value)

        self.gray_img = target

    def _remove_background(self):
        buf = cv2.blur(self.gray_img, (3, 3))
        buf = cv2.threshold(buf, 36, 255, cv2.THRESH_BINARY)[1]
        contours = []
        contours, _ = cv2.findContours(buf, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE, contours=contours)

        max_size = 0
        max_contour = None
        for contour in contours:
            size = cv2.contourArea(contour)
            if size > max_size:
                max_size = size
                max_contour = contour

        mask = np.zeros((buf.shape[0], buf.shape[1], 1), dtype=np.uint8)
        mask = cv2.drawContours(mask, [max_contour], -1, 1, cv2.FILLED)

        self.gray_img = cv2.multiply(self.gray_img, mask)

        return max_contour


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


def read_data(base_path):
    cr = contourreader.ContourReader()
    mrts = []
    for mrt_dir in Path.iterdir(base_path):
        mrt = MRT(mrt_dir)

        for ima_path in mrt_dir.glob('*.IMA'):
            name = ima_path.stem + '.txt'
            contour_path = mrt_dir.joinpath('Save/Autosave').joinpath(name)
            layer = Layer(ima_path, contour_path,cr)
            mrt.layers.append(layer)

        mrts.append(mrt)

    return mrts
