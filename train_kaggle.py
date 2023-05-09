import kaggle
import json
import os

def run_script(script: str, data=[], util_scripts=[], gpu=False, is_util=False):
    # create a list of kernels (secondary scripts) using the filenames in util_scripts
    kernels = [f"{username}/{script}" for script in util_scripts]
    # kernel metadata
    metadata = {
        "id": f"{username}/{script}",
        "title": script,
        "code_file": f"{script}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "keywords": ["util-script"] if is_util else [],
        "enable_gpu": gpu,
        "enable_internet": "true",
        "dataset_sources": data,
        "competition_sources": [],
        "kernel_sources": kernels,
    }
    with open("kernel-metadata.json", "w") as file:
        json.dump(metadata, file)
    # wait for the secondary scripts to run before running the main script
    if util_scripts:
        print("\nWaiting for secondary scripts to finish running.")
        complete = []
        while sum(complete) != len(kernels):
            status = [api.kernels_status(kernel)["status"] for kernel in kernels]
            complete = ["complete" in st for st in status]
            failed = ["error" in st for st in status]
            if sum(failed) > 0:
                print(
                    "Scripts failed:",
                    ",".join([util_scripts[idx] for idx, fail in enumerate(failed) if fail]),
                )
                return
    api.kernels_push_cli(os.getcwd())  # push kernel to kaggle
    os.remove("kernel-metadata.json")


if __name__ == "__main__":
    api = kaggle.api
    api.authenticate()
    username = api.get_config_value(api.CONFIG_NAME_USER)
    datasets = [
        "davidfmora/deepglobe-land-cover-classification-procesed"
    ]

    run_script("dataset", data=datasets, is_util=True)
    run_script("model", is_util=True)
    run_script("utils", is_util=True)
    run_script("train", util_scripts=["dataset", "model", "utils"], data=datasets, gpu=True)
