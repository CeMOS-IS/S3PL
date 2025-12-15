import os
import json
import argparse
from train import train

def load_config(config_path, args):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Override config values if provided via CLI
    if args.data_dir is not None: config["data_dir"] = args.data_dir
    if args.number_classes is not None: config["number_classes"] = args.number_classes
    if args.eval_picking is not None: config["evaluate_peak_picking"] = args.eval_picking
    if args.number_peaks is not None: config["number_peaks"] = args.number_peaks
    if args.peaks_per_spectral_patch is not None: config["peaks_per_spectral_patch"] = args.peaks_per_spectral_patch
    if args.spectral_patch_size is not None: config["spectral_patch_size"] = args.spectral_patch_size
    if args.kernel_depth_d1 is not None: config["kernel_depth_d1"] = args.kernel_depth_d1
    if args.kernel_depth_d2 is not None: config["kernel_depth_d2"] = args.kernel_depth_d2
    if args.n_epochs is not None: config["n_epochs"] = args.n_epochs
    if args.batch_size is not None: config["batch_size"] = args.batch_size
    if args.learning_rate is not None: config["learning_rate"] = args.learning_rate
    if args.dropout is not None: config["dropout"] = args.dropout
    if args.random_seed is not None: config["random_seed"] = args.random_seed

    return config

if __name__ == "__main__":
    directory_name = os.path.dirname(__file__)

    parser = argparse.ArgumentParser("S3PL training, peak picking and evaluation.")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--number_classes", type=int, default=None, help="number of annotated classes in the segmentation mask (including background)")
    parser.add_argument("--eval_picking", type=bool, default=True, help="whether picked peaks should be evaluated (segmentation is required)")
    parser.add_argument("--number_peaks", type=int, default=None, required=False)
    parser.add_argument("--peaks_per_spectral_patch", type=int, default=512)
    parser.add_argument("--spectral_patch_size", type=int, default=3)
    parser.add_argument("--kernel_depth_d1", type=int, default=51)
    parser.add_argument("--kernel_depth_d2", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--random_seed", type=int, default=1)

    args = parser.parse_args()
    config = load_config(config_path=f'{directory_name}/config.json', args=args)

    mSCF1 = train(config)