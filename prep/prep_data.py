from pathlib import Path

import cv2
import numpy as np
import pydicom

import prep.contourreader as contourreader


class MRT:

    def __init__(self, path):
        self.path = path
        self.layers = []


class Layer:
    def __init__(self, mrt_path, contours_path, contour_reader):
        self.mrt_path = mrt_path
        self.contours = contours_path
        self.gray_img = None
        self.contour_reader = contour_reader
        self.name = mrt_path.stem
        self.label_image = None

    def process(self):
        self._convert_data()
        mask = self._remove_background()
        input_contour = self.contour_reader.parseFile(self.contours)
        self._generate_label_image(mask, input_contour)

    def _convert_data(self):
        dataset = pydicom.dcmread(self.mrt_path)
        window_center = dataset[0x0028, 0x1050]
        window_width = dataset[0x0028, 0x1051]

        w_center = window_center.value
        w_width = window_width.value
        small_mask = dataset.pixel_array < w_center - 0.5 - ((w_width - 1) / 2)
        big_mask = dataset.pixel_array > w_center - 0.5 + ((w_width - 1) / 2)
        target = (((dataset.pixel_array - (w_center - 0.5)) / (w_width - 1)) + 0.5) * (255 - 0) + 0
        target[small_mask] = 0
        target[big_mask] = 255
        target = target.astype(np.uint8)

        self.gray_img = target

    def _remove_background(self):
        buf = cv2.blur(self.gray_img, (5, 5))
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

        mask = np.zeros((buf.shape[0], buf.shape[1]), dtype=np.uint8)
        mask = cv2.drawContours(mask, [max_contour], -1, 1, cv2.FILLED)

        hull = cv2.convexHull(max_contour)

        convex_mask = np.zeros((self.gray_img.shape[0], self.gray_img.shape[1]), dtype=np.uint8)
        convex_mask = cv2.drawContours(convex_mask, [hull], -1, 1, cv2.FILLED)
        diff = convex_mask - mask

        max_bone_size = -1
        max_bone = None
        for contour in cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            size = cv2.contourArea(contour)
            if size > max_bone_size:
                max_bone_size = size
                max_bone = contour

        bone_img = np.zeros(diff.shape, dtype=np.uint8)
        bone_img = cv2.drawContours(bone_img, [max_bone], -1, 1, cv2.FILLED)

        mask += bone_img
        self.gray_img = cv2.multiply(self.gray_img, mask)
        return mask

    def _generate_label_image(self, mask, contours):
        res = mask
        for i, contour in enumerate(contours):
            res = cv2.drawContours(res, [contour], -1, i + 2, cv2.FILLED)

        self.label_image = res


def prepare_data(dicom_path, output_path):
    create_structure(output_path)

    print('prepare data')
    mrts = read_data(dicom_path)
    for mrt in mrts:
        img_dir = output_path.joinpath('images', mrt.path.name)
        lab_dir = output_path.joinpath('labels', mrt.path.name)
        col_dir = output_path.joinpath('colors', mrt.path.name)

        img_dir.mkdir(exist_ok = True)
        lab_dir.mkdir(exist_ok = True)
        col_dir.mkdir(exist_ok = True)

        for layer in mrt.layers:
            layer.process()
            cv2.imwrite(str(img_dir.joinpath(layer.name + '.png')), layer.gray_img)
            cv2.imwrite(str(lab_dir.joinpath(layer.name + '.png')), layer.label_image)
            img_color = color_img_from_labels(layer.label_image)
            cv2.imwrite(str(col_dir.joinpath(layer.name + '.png')), img_color)


def create_structure(output_path):
    print('creating project structure')
    image_path = output_path.joinpath(Path("images"))
    label_path = output_path.joinpath(Path("labels"))
    color_path = output_path.joinpath(Path("colors"))

    output_path.mkdir(exist_ok=True)
    image_path.mkdir(exist_ok=True)
    label_path.mkdir(exist_ok=True)
    color_path.mkdir(exist_ok=True)


def color_img_from_labels(img_labels):
    np_max = np.max(img_labels) + 3
    step = np.uint8(360 / np_max)
    label_hue = np.uint8(img_labels * step)
    blank_ch = 255 * np.ones_like(label_hue)
    img_color = cv2.merge([label_hue, blank_ch, blank_ch])

    img_color = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)

    img_color[label_hue == 0] = 0
    img_color[label_hue == step] = 255

    return img_color


def read_data(base_path):
    contour_reader = contourreader.ContourReader()
    mrts = []
    for mrt_dir in Path.iterdir(base_path):
        mrt = MRT(mrt_dir)

        for ima_path in mrt_dir.glob('*.IMA'):
            name = ima_path.stem + '.txt'
            contour_path = mrt_dir.joinpath('Save/Autosave').joinpath(name)
            if contour_path.exists():
                layer = Layer(ima_path, contour_path, contour_reader)
                mrt.layers.append(layer)

        mrts.append(mrt)

    return mrts
