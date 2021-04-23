import os
import tarfile


def main():
    with tarfile.open(os.path.join("data", "imagenette2-320.tgz")) as f:
        f.extractall("data")


if __name__ == '__main__':
    main()
