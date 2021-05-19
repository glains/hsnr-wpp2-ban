from pathlib import Path


def prepare_data(dicom_path, output_path):
    create_structure(output_path)
    print('prepare data')


def create_structure(output_path):
    print('creating project structure')
    image_path = output_path.joinpath(Path("images"))
    label_path = output_path.joinpath(Path("labels"))
    color_path = output_path.joinpath(Path("colors"))

    output_path.mkdir(exist_ok=True)
    image_path.mkdir(exist_ok=True)
    label_path.mkdir(exist_ok=True)
    color_path.mkdir(exist_ok=True)
