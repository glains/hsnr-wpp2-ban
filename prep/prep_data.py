from pathlib import Path


def prepare_data():
    create_structure()
    print('prepare data')


def create_structure():
    print('creating project structure')
    Path("../images").mkdir(exist_ok=True)
    Path('../labels').mkdir(exist_ok=True)
    Path('../colors').mkdir(exist_ok=True)
