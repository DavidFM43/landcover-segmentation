import kaggle
import json
import os
from pathlib import Path

api = kaggle.api
api.authenticate()
username = api.get_config_value(api.CONFIG_NAME_USER)
src_dir = Path("src")
datasets = [
    "balraj98/deepglobe-land-cover-classification-dataset",
    "davidfmora/processed-masks",
]
util_scripts = ["dataset", "model", "utils"]


def run_script(script, data=[], scripts=[], gpu=False, util=False):
    kernels = [f"{username}/{script}" for script in scripts]
    # kernel metadata
    metadata = {
        "id": f"{username}/{script}",
        "title": script,
        "code_file": f"{src_dir / script}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "keywords": ["util-script"] if util else [],
        "enable_gpu": gpu,
        "enable_internet": "true",
        "dataset_sources": data,
        "competition_sources": [],
        "kernel_sources": kernels,
    }
    with open("kernel-metadata.json", "w") as file:
        json.dump(metadata, file)
    # wait for the secondary scripts to run before running the main script
    if scripts:
        print("\nWaiting for secondary scripts to finish running.")
        complete = []
        while sum(complete) != len(kernels):
            status = [api.kernels_status(kernel)["status"] for kernel in kernels]
            complete = ["complete" in st for st in status]
            failed = ["error" in st for st in status]
            if sum(failed) > 0:
                print(
                    "Scripts failed:",
                    ",".join([scripts[idx] for idx, fail in enumerate(failed) if fail]),
                )
                return
    api.kernels_push_cli(os.getcwd())  # push kernel to kaggle
    os.remove("kernel-metadata.json")


if __name__ == "__main__":
    run_script("dataset", data=datasets, util=True)
    run_script("model", util=True)
    run_script("utils", util=True)
    run_script("train", scripts=util_scripts, data=datasets, gpu=True)
