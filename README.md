# Inference with Happyrobot API

## Installation

Create a conda environment

    conda create -n happyrobot python=3.9
    conda activate happyrobot

Install dependencies

    pip install -r requirements.txt


## Usage

    python run_inference.py <PATH_TO_IMAGE_OR_FOLDER> --endpoint <ENDPOINT> --apikey <API_KEY> [--visualize] [--save_images] [--output_folder <OUTPUT_FOLDER>] [--save_json]

### Arguments

- `--visualize`: flag to show results in a window.
- `--save_images`: flag to save results to disk.
- `--output_folder`: path to folder where to save results.
- `--save_json`: flag to save results to disk in json format.