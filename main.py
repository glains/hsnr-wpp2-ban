from prep.prep_data import prepare_data
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', type=Path, action='store', dest='inputPath')
    parser.add_argument('-o', type=Path, action='store', dest='outputPath')

    args = parser.parse_args()

    prepare_data(args.inputPath, args.outputPath)


if __name__ == "__main__":
    # execute only if run as a script
    main()
