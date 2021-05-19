from pathlib import Path
import os


def prepare_data(dicomPath, outputPath):
    create_structure(outputPath)
    print('prepare data')


def create_structure(outputPath):
    print('creating project structure')
    imagePath = outputPath.joinpath(Path("images"))
    labelPath = outputPath.joinpath(Path("labels"))
    colorPath = outputPath.joinpath(Path("colors"))
    
    outputPath.mkdir(exist_ok = True)
    imagePath.mkdir(exist_ok = True)
    labelPath.mkdir(exist_ok = True)
    colorPath.mkdir(exist_ok = True)
