import shutil
import time
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve
import appdirs


def reporthook(count, block_size, total_size, msg=None):
    global start_time
    if msg is not None:
        print(msg)
        return
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


def downloadFiles(url, target_folder=None, progress_callback=None):
    file = Path(url).name
    target = Path(file)
    output_folder = Path(target.stem)
    if target_folder is not None:
        output_folder = Path(target_folder) / output_folder
    file_download_path = str(Path(output_folder).parent / file)
    if progress_callback is None:
        progress_callback = reporthook
    if not output_folder.exists():
        reporthook(None, None, None, msg="Downloading File")
        Path(output_folder).parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(url), file_download_path, progress_callback)
        reporthook(None, None, None, msg="unzipping...")
        shutil.unpack_archive(file_download_path, output_folder.parent)
        os.remove(file_download_path)


def loadExample(name, target_folder=None, progress_callback=None):
    if target_folder is None:
        target_folder = appdirs.user_data_dir("saenopy", "rgerum")
    if name == "ClassicSingleCellTFM":
        downloadFiles("https://github.com/rgerum/saenopy/releases/download/v0.7.4/1_ClassicSingleCellTFM.zip", target_folder, progress_callback=progress_callback)
    if name == "DynamicalSingleCellTFM":
        downloadFiles("https://github.com/rgerum/saenopy/releases/download/v0.7.4/2_DynamicalSingleCellTFM.zip", target_folder, progress_callback=progress_callback)



def getExamples():
    example_path = Path(appdirs.user_data_dir("saenopy", "rgerum"))
    image_path = Path(__file__).parent / "img" / "examples"
    return {
        "ClassicSingleCellTFM": {
            "desc": "Hepatic stellate cells in 1.2mg/ml collagen with relaxed and deformed stacks.\nRelaxed state induced with cytochalasin D.\n3 examples",
            "img": image_path / "example1.png",
            "voxel_size": [0.7211, 0.7211, 0.988],
            "stack": example_path / '1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
            "reference_stack": example_path / '1_ClassicSingleCellTFM/Relaxed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
            "output_path": example_path / '1_ClassicSingleCellTFM/example_output',
        },
        "DynamicalSingleCellTFM": {
            "desc": "Single natural killer cell that migrated through 1.2mg/ml collagen, recorded for 24min.\n1 example",
            "img": image_path / "example2.png",
            "voxel_size": [0.2407, 0.2407, 1.0071],
            "time_delta": 60,
            "stack": example_path / '2_DynamicalSingleCellTFM/data/Pos*_S001_t{t}_z{z}_ch{c:00}.tif',
            "output_path": example_path / '2_DynamicalSingleCellTFM/example_output',
        },
    }
