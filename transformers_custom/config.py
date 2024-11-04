import os

def get_config():
    config = {
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 1e-4,
    "seq_len": 350,
    "d_model": 512,
    "lang_src": "en",
    "lang_tgt": "it",
    "model_folder": "weights",
    "model_basename": "custom_transformer__",
    "preload_model": None,
    "tokenizer_file": "tokenizer_{0}.json",
    "experiment_name": "runs/custom_transformer"
    }

    return config

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join(model_folder, model_filename)




