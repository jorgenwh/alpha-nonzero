import argparse
import os
from anz.selfplay import SPConfig, SelfPlayManager

if __name__ == "__main__":
    config = SPConfig()

    arg_parser = argparse.ArgumentParser()
    args = arg_parser.parse_args()

    assert config.output_dir is not None

    if not os.path.isdir(config.output_dir):
        os.mkdir(config.output_dir)

    manager = SelfPlayManager(config)
    manager.start()
