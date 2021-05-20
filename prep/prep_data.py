from pathlib import Path

import numpy as np
import pydicom
import time

import cv2

class MRT:

    def __init__(self, path):
        self.path = path
        self.layers = []


class Layer:
    def __init__(self, mrt_path, contours_path):
        self.mrt_path = mrt_path
        self.contours = contours_path
        self.gray_img = None
        self.name = mrt_path.stem

    def process(self):
        self._convert_data()
        mask = self._remove_background()
        #self._generate_label_image()

    def _convert_data(self):
        ds = pydicom.dcmread(self.mrt_path)
        window_center = ds[0x0028, 0x1050]
        window_width = ds[0x0028, 0x1051]

        wCenter = window_center.value
        wWidth = window_width.value
        smallMask = ds.pixel_array < wCenter - 0.5 - ((wWidth - 1) / 2)
        bigMask = ds.pixel_array >  wCenter - 0.5 + ((wWidth - 1) / 2)
        target = (((ds.pixel_array - (wCenter - 0.5)) / (wWidth - 1)) + 0.5) * (255-0) + 0
        target[smallMask] = 0
        target[bigMask] = 255
        target = target.astype(np.uint8)

        self.gray_img = target
    
    def _remove_background(self):
        buf = cv2.blur(self.gray_img,(5, 5))
        buf = cv2.threshold(buf, 36, 255, cv2.THRESH_BINARY)[1]
        contours = []
        contours, _ = cv2.findContours(buf,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE, contours=contours)

        maxSize = 0
        maxContour = None
        for contour in contours:
            size = cv2.contourArea(contour)
            if size > maxSize:
                maxSize = size
                maxContour = contour
        
        mask = np.zeros((buf.shape[0], buf.shape[1], 1), dtype=np.uint8)
        mask = cv2.drawContours(mask, [maxContour], -1, 1, cv2.FILLED)
        

        hull = cv2.convexHull(maxContour)

        convexMask = np.zeros((self.gray_img.shape[0], self.gray_img.shape[1], 1), dtype=np.uint8)
        convexMask = cv2.drawContours(convexMask, [hull], -1, 1, cv2.FILLED)
        diff = convexMask - mask       

        maxBoneSize = -1
        maxBone = None
        for contour in cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            size = cv2.contourArea(contour)
            if size > maxBoneSize:
                maxBoneSize = size
                maxBone = contour

        boneImg = np.zeros(diff.shape, dtype=np.uint8)
        boneImg = cv2.drawContours(boneImg, [maxBone], -1, 1, cv2.FILLED)
        
        mask += boneImg
        self.gray_img = cv2.multiply(self.gray_img, mask)
        return mask

def _generate_label_image(self, mask, contours):
    pass

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
    mrts = []
    for mrt_dir in Path.iterdir(base_path):
        mrt = MRT(mrt_dir)

        for ima_path in mrt_dir.glob('*.IMA'):
            name = ima_path.stem + '.txt'
            contour_path = mrt_dir.joinpath('Save/Autosave').joinpath(name)
            if contour_path.exists():
                layer = Layer(ima_path, contour_path)
                mrt.layers.append(layer)

        mrts.append(mrt)

    return mrts
