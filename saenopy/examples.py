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
    if name == "OrganoidTFM":
        downloadFiles("https://github.com/rgerum/saenopy/releases/download/v0.7.4/4_OrganoidTFM.zip", target_folder, progress_callback=progress_callback)
    


def getExamples():
    example_path = Path(appdirs.user_data_dir("saenopy", "rgerum"))
    image_path = Path(__file__).parent / "img" / "thumbnails"
    return {
        "ClassicSingleCellTFM": {
            "desc": "Hepatic stellate cells in 1.2mg/ml collagen with relaxed and deformed stacks.\nRelaxed state induced with cytochalasin D.\n3 examples",
            "img": image_path / "liver_fibroblast_icon.png",
            "voxel_size": [0.7211, 0.7211, 0.988],
            "stack": example_path / '1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
            "reference_stack": example_path / '1_ClassicSingleCellTFM/Relaxed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
            "output_path": example_path / '1_ClassicSingleCellTFM/example_output',
            "piv_parameter": {'win_um': 35.0, 'elementsize': 14.0, 'signoise_filter': 1.3, 'drift_correction': True},
            "interpolate_parameter": {'reference_stack': 'first', 'element_size': 14.0, 'inner_region': 200.0, 'thinning_factor': 0, 'mesh_size_same': True, 'mesh_size_x': 200.0, 'mesh_size_y': 200.0, 'mesh_size_z': 200.0},
            "solve_parameter": {'k': 6062.0, 'd0': 0.0025, 'lambda_s': 0.0804, 'ds':  0.034, 'alpha': 10**10, 'stepper': 0.33, 'i_max': 400, 'rel_conv_crit': 0.009},
        },
        "DynamicalSingleCellTFM": {
            "desc": "Single natural killer cell that migrated through 1.2mg/ml collagen, recorded for 24min.\n1 example",
            "img": image_path / "Dynamic_icon.png",
            "voxel_size": [0.2407, 0.2407, 1.0071],
            "time_delta": 60,
            "stack": example_path / '2_DynamicalSingleCellTFM/data/Pos*_S001_t{t}_z{z}_ch{c:00}.tif',
            "output_path": example_path / '2_DynamicalSingleCellTFM/example_output',
            "piv_parameter": {'win_um': 12.0, 'elementsize': 4.0, 'signoise_filter': 1.3, 'drift_correction': True},
            "interpolate_parameter": {'reference_stack': 'median', 'element_size': 4.0, 'inner_region': 100.0, 'thinning_factor': 0, 'mesh_size_same': True, 'mesh_size_x': 200.0, 'mesh_size_y': 200.0, 'mesh_size_z': 200.0},
            "solve_parameter": {'k': 1449.0, 'd0': 0.0022, 'lambda_s': 0.032, 'ds': 0.055, 'alpha':  10**10, 'stepper': 0.33, 'i_max': 100},
            "crop": {"z": (20, -20)},
        },
        "OrganoidTFM": {
            "desc": "Intestinal organoid in 1.2mg/ml collagen",
            "img": image_path / "StainedOrganoid_icon.png",
            "voxel_size": [1.444, 1.444, 1.976],
            "stack": example_path / '4_OrganoidTFM/Pos007_S001_t50_z{z}_ch00.tif',
            "reference_stack": example_path / '4_OrganoidTFM/Pos007_S001_t6_z{z}_ch00.tif',
            "output_path": example_path / '4_OrganoidTFM/example_output',
            "piv_parameter": {'win_um': 40.0, 'elementsize': 30.0, 'signoise_filter': 1.3, 'drift_correction': True},
            "interpolate_parameter": {'reference_stack': 'first', 'element_size': 30, 'inner_region': 100.0, 'thinning_factor': 0, 'mesh_size_same': False, 'mesh_size_x': 900.0, 'mesh_size_y': 900.0, 'mesh_size_z': 900.0},
            "solve_parameter": {'k': 6062.0, 'd0': 0.0025, 'lambda_s': 0.0804, 'ds':  0.034, 'alpha':  10**10, 'stepper': 0.33, 'i_max': 500,  'rel_conv_crit': 0.00003},
    }
    

    }
