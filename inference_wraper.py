import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import sys
import os

import torch
import torchaudio
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from src import metrics, commons
from src.models import models
from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.in_the_wild_dataset import InTheWildDataset
from src.datasets.asvspoof_dataset import ASVSpoof2019DatasetOriginal #TODO: Use
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset #TODO: Use
#TODO: Add our dataset here for eval

def get_data(datasets_paths:str, sr: int) -> torch.Tensor:
    """
    Load audio from the specified path, resample it to the given sample rate (sr),
    and return it as a PyTorch tensor.

    Args:
        path (str): Path to the audio file.
        sr (int): Target sample rate for resampling.

    Returns:
        torch.Tensor: Tensor containing the audio data.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(datasets_paths)
    print(sample_rate)
    # Resample audio if needed
    if sample_rate != sr and sr != None:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)

    return waveform


def main(
    model_paths: List[Path],
    model_config: Dict,
    device: str,
):
    model_name, model_parameters = model_config["name"], model_config["parameters"]

    # Load model architecture
    model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    if len(model_paths):
        model.load_state_dict(torch.load(model_paths))
    model = model.to(device)

    audio = get_data(
        datasets_paths=args.data_path,
        sr=args.sr,
    )


    model.eval()
    with torch.no_grad():
        audio = audio.to(device)

        audio_pred = model(audio).squeeze(1)
        audio_pred = torch.sigmoid(audio_pred)
        audio_pred_label = (audio_pred + 0.5).int()


    return audio_pred_label


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # If assigned as None, then it won't be taken into account
    IN_THE_WILD_DATASET_PATH = "../in-the-wild-dataset/release_in_the_wild"

    parser.add_argument(
        "--data_path", type=str
    )

    parser.add_argument(
        "--sr", type=int, default=16000
    )

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )

   

    parser.add_argument("--cpu", "-c", help="Force using cpu", action="store_true")

    args = parser.parse_args()

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds - this should not actually change anything
    commons.set_seed(seed)
    
    human_or_fake = main(
        model_paths=config["checkpoint"].get("path", []),
        model_config=config["model"],
        device=device,
    )

    

    print("##### Audio is: ", "Human" if human_or_fake.item()==0 else "Fake")