import shutil
import time
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration + 0.001))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def downloadFiles(url):
    file = Path(url).name
    target = Path(file)
    output_folder = Path(target.stem)
    if not output_folder.exists():
        print("Downloading File", file)
        urlretrieve(str(url), str(file), reporthook)
        print("unzipping...")
        shutil.unpack_archive(str(file), ".")
        os.remove(str(file))

def loadExample(name):
    if name == "ClassicSingleCellTFM":
        downloadFiles("https://github.com/rgerum/saenopy/releases/download/v0.7.4/1_ClassicSingleCellTFM.zip")
    if name == "DynamicalSingleCellTFM":
        downloadFiles("https://github.com/rgerum/saenopy/releases/download/v0.7.4/2_DynamicalSingleCellTFM.zip")

