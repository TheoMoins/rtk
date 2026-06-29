
from rtk.task import DownloadIIIFImageTask, KrakenAltoCleanUpCommand, ClearFileCommand, \
    DownloadIIIFManifestTask, YALTAiCommand, KrakenRecognizerCommand, ExtractZoneAltoCommand
from rtk import utils


path = "/home/tmoins/Documents/WPC1/"

batches = utils.batchify_textfile(path+"data/manifests_test_french/manifests_test_french.txt", batch_size=4)
from re import sub

import torch
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import random
random.seed(0)

def kebab(s):
    return sub(r"https?-", "", '-'.join(
        sub(r"(\W+)"," ",
        sub(r"[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+",
        lambda mo: ' ' + mo.group(0).lower(), s)).split()
    ))


for batch in batches:
    # Download Manifests
    print("[Task] Download manifests")
    dl = DownloadIIIFManifestTask(
        batch,
        output_directory="data/manifests_test_french/",
        naming_function=lambda x: kebab(x), multiprocess=10
    )
    dl.process()

    # Download Files
    print("[Task] Download JPG")
    dl = DownloadIIIFImageTask(
        dl.output_files,
        max_height=2500,
        multiprocess=4,
        downstream_check=DownloadIIIFImageTask.check_downstream_task("xml", utils.check_content)
    )
    dl.process()


