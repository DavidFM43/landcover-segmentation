import kaggle
import json
import os


def run_kernel(kernel_name: str, input_data=[], is_nb=True):
    """Run a kaggle kernel"""

    kernel_type = "notebook" if is_nb else "script"
    ext = "ipynb" if is_nb else "py"
    # kernel metadata
    metadata = {
        "id": f"{username}/{kernel_name}",
        "title": kernel_name,
        "code_file": f"{kernel_name}.{ext}",
        "language": "python",
        "kernel_type": kernel_type,
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": input_data,
        "competition_sources": [],
        "keywords": ["gpu"],
    }
    # write metadata to json file
    with open("kernel-metadata.json", "w") as file:
        json.dump(metadata, file)

    api.kernels_push_cli(os.getcwd())  # push kernel to kaggle
    os.remove("kernel-metadata.json")


if __name__ == "__main__":
    api = kaggle.api
    api.authenticate()
    username = api.get_config_value(api.CONFIG_NAME_USER)

    input_data = ["davidfmora/deepglobe-land-cover-classification-procesed"]
    run_kernel("deepglobe-train", input_data=input_data)
