import os
import json
from train import train

if __name__ == "__main__":
    directory_name = os.path.dirname(__file__)
    with open(f'{directory_name}/config.json') as f:
        config = json.load(f)

    mSCF1 = train(config)