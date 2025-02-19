import os
import torch


def get_output_folder(parent_dir, dataset_name):
    """
    Return save folder.

    Assumes folders in the parent_dir have suffix _run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if (
            not os.path.isdir(os.path.join(parent_dir, folder_name))
            or dataset_name not in folder_name
        ):
            continue
        try:
            folder_name = int(folder_name.split("_run")[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, dataset_name)
    parent_dir = parent_dir + "_run{}".format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def load_dataset(path):
    train = torch.load(f"{path}/train.pt")
    train_labels = torch.load(f"{path}/train_labels.pt")
    test = torch.load(f"{path}/test.pt")
    test_labels = torch.load(f"{path}/test_labels.pt")
    return train, train_labels, test, test_labels