from pathlib import Path

import numpy as np
import pydicom

import cv2

import prep.contourreader as contourreader

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
        self.name = mrt_path.stem

    def process(self):
        self._convert_data()
        mask = self._remove_background()
        input_contour = self.contour_reader.parseFile(self.contours)
        self._generate_label_image(mask, input_contour)

    def _convert_data(self):
        ds = pydicom.dcmread(self.mrt_path)
        window_center = ds[0x0028, 0x1050]
        window_width = ds[0x0028, 0x1051]

        w_center = window_center.value
        w_width = window_width.value
        smallMask = ds.pixel_array < w_center - 0.5 - ((w_width - 1) / 2)
        bigMask = ds.pixel_array >  w_center - 0.5 + ((w_width - 1) / 2)
        target = (((ds.pixel_array - (w_center - 0.5)) / (w_width - 1)) + 0.5) * (255-0) + 0
        target[smallMask] = 0
        target[bigMask] = 255
        target = target.astype(np.uint8)

        self.gray_img = target

    def _remove_background(self):
        buf = cv2.blur(self.gray_img,(5, 5))
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
        

        hull = cv2.convexHull(max_contour)

        convex_mask = np.zeros((self.gray_img.shape[0], self.gray_img.shape[1], 1), dtype=np.uint8)
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
        for i,contour in enumerate(contours):
            res = cv2.drawContours(res, [contour], -1, i+2, cv2.FILLED)
        test = None
        test = cv2.normalize(res, test,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        test = (test*255).astype(np.uint8)

        self.label_image = res

def prepare_data(dicom_path, output_path):
    create_structure(output_path)

    print('prepare data')
    mrts = read_data(dicom_path)
    for mrt in mrts:
        outdir = output_path.joinpath('images', mrt.path.name)
        outdir.mkdir(exist_ok=True)
        for layer in mrt.layers:
            layer.process()
            cv2.imwrite(str(outdir.joinpath(layer.name+'.png')), layer.gray_img)

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
            if contour_path.exists():
                layer = Layer(ima_path, contour_path,cr)
                mrt.layers.append(layer)

        mrts.append(mrt)

    return mrts
