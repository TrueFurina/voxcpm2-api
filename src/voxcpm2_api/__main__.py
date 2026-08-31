from multiprocessing import freeze_support

from voxcpm2_api.cli import main


if __name__ == "__main__":
    freeze_support()
    main()
