# Inference with Happyrobot API

## Installation

Create a conda environment

    conda create -n happyrobot python=3.9
    conda activate happyrobot

Install dependencies

    pip install -r requirements.txt


## Usage

    python run_inference.py <PATH_TO_IMAGE_OR_FOLDER> --endpoint <ENDPOINT> --apikey <API_KEY>

You can use the `--visualize` flag to show results in a window.

    python run_inference.py <PATH_TO_IMAGE_OR_FOLDER> --endpoint <ENDPOINT> --apikey <API_KEY> --visualize

Or you can use the `--save` flag to save results to disk. If you do so, you can optionally define the `--output_folder` where you'd like to save results.