import shutil
import time
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve
import appdirs
from saenopy.gui.common.resources import resource_path


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


def download_files(url, target_folder=None, progress_callback=None):
    file = Path(url).name
    target = Path(file)
    output_folder = Path(target.stem)
    if target_folder is not None:
        output_folder = Path(target_folder) / output_folder
    file_download_path = str(Path(output_folder).parent / file)
    if progress_callback is None:
        progress_callback = reporthook
    if not output_folder.exists():
        progress_callback(None, None, None, msg="Downloading File")
        Path(output_folder).parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(str(url), file_download_path, progress_callback)
        progress_callback(None, None, None, msg="unzipping...")
        shutil.unpack_archive(file_download_path, output_folder.parent)
        os.remove(file_download_path)


def load_example(name, target_folder=None, progress_callback=None, evaluated=False):
    if target_folder is None:
        target_folder = appdirs.user_data_dir("saenopy", "rgerum")
    example = get_examples()[name]
    url = example["url"]
    download_files(url, target_folder, progress_callback=progress_callback)

    if evaluated:
        evaluated_folder = Path(target_folder) / Path(Path(url).name).stem / "example_output"
        if not (evaluated_folder / example["url_evaluated_file"][0]).exists():
            download_files(example["url_evaluated"], evaluated_folder, progress_callback=progress_callback)
        return [evaluated_folder / file for file in example["url_evaluated_file"]]


def get_examples():
    example_path = Path(appdirs.user_data_dir("saenopy", "rgerum"))
    image_path = Path(resource_path("thumbnails"))
    return {
        "ClassicSingleCellTFM": {
            "desc": "Hepatic stellate cells in 1.2mg/ml collagen with relaxed and deformed stacks.\nRelaxed state induced with cytochalasin D.\n3 examples",
            "img": image_path / "liver_fibroblast_icon.png",
            "voxel_size": [0.7211, 0.7211, 0.988],
            "stack": example_path / '1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
            "reference_stack": example_path / '1_ClassicSingleCellTFM/Relaxed/Mark_and_Find_001/Pos*_S001_z{z}_ch{c:00}.tif',
            "output_path": example_path / '1_ClassicSingleCellTFM/example_output',
            "piv_parameters": {'window_size': 35.0, 'element_size': 14.0, 'signal_to_noise': 1.3, 'drift_correction': True},
            "mesh_parameters": {'reference_stack': 'first', 'element_size': 14.0, 'mesh_size': 'piv'},
            "material_parameters": {'k': 6062.0, 'd_0': 0.0025, 'lambda_s': 0.0804, 'd_s':  0.034},
            "solve_parameters": {'alpha': 10**10, 'step_size': 0.33, 'max_iterations': 400, 'rel_conv_crit': 0.009},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/1_ClassicSingleCellTFM.zip",
            "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/1_ClassicSingleCellTFM_evaluated.zip",
            "url_evaluated_file": ["Pos004_S001_z{z}_ch{c00}.saenopy", "Pos007_S001_z{z}_ch{c00}.saenopy", "Pos008_S001_z{z}_ch{c00}.saenopy"],
        },
        "DynamicalSingleCellTFM": {
            "desc": "Single natural killer cell that migrated through 1.2mg/ml collagen, recorded for 24min.\n1 example",
            "img": image_path / "Dynamic_icon.png",
            "voxel_size": [0.2407, 0.2407, 1.0071],
            "time_delta": 60,
            "stack": example_path / '2_DynamicalSingleCellTFM/data/Pos*_S001_t{t}_z{z}_ch{c:00}.tif',
            "output_path": example_path / '2_DynamicalSingleCellTFM/example_output',
            "piv_parameters": {'window_size': 12.0, 'element_size': 4.0, 'signal_to_noise': 1.3, 'drift_correction': True},
            "mesh_parameters": {'reference_stack': 'median', 'element_size': 4.0, 'mesh_size': 'piv'},
            "material_parameters": {'k': 1449.0, 'd_0': 0.0022, 'lambda_s': 0.032, 'd_s': 0.055},
            "solve_parameters": {'alpha': 10**10, 'step_size': 0.33, 'max_iterations': 100},
            "crop": {"z": (20, -20)},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/2_DynamicalSingleCellTFM.zip",
            "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/2_DynamicalSingleCellTFM_evaluated.zip",
            "url_evaluated_file": ["Pos002_S001_t{t}_z{z}_ch{c00}.saenopy"],
        },
        "OrganoidTFM": {
            "desc": "Intestinal organoid in 1.2mg/ml collagen",
            "img": image_path / "StainedOrganoid_icon.png",
            "voxel_size": [1.444, 1.444, 1.976],
            "stack": example_path / '4_OrganoidTFM/Pos007_S001_t50_z{z}_ch00.tif',
            "reference_stack": example_path / '4_OrganoidTFM/Pos007_S001_t6_z{z}_ch00.tif',
            "output_path": example_path / '4_OrganoidTFM/example_output',
            "piv_parameters": {'window_size': 40.0, 'element_size': 30.0, 'signal_to_noise': 1.3, 'drift_correction': True},
            "mesh_parameters": {'reference_stack': 'first', 'element_size': 30, 'mesh_size': (738.0, 738.0, 738.0)},
            "material_parameters": {'k': 6062.0, 'd_0': 0.0025, 'lambda_s': 0.0804, 'd_s': 0.034},
            "solve_parameters": {'alpha':  10**10, 'step_size': 0.33, 'max_iterations': 1400,  'rel_conv_crit': 1e-7},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/4_OrganoidTFM.zip",
            "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/4_OrganoidTFM_evaluated.zip",
            "url_evaluated_file": ["Pos007_S001_t50_z{z}_ch00.saenopy"],
        },
        "BrightfieldTFM": {
            "desc": "Traction forces around an immune cell in collagen 1.2mg/ml calculated on simple brightfield images",
            "img": image_path / "BFTFM_2.png",
            "voxel_size": [0.15, 0.15, 2.0],
            "crop": {'x': (1590, 2390), 'y': (878, 1678), 'z': (30, 90)},
            "stack": example_path / '6_BrightfieldNK92Data/2023_02_14_12_0920_stack.tif[z]',
            "reference_stack": example_path / '6_BrightfieldNK92Data/2023_02_14_12_0850_stack.tif[z]',
            "output_path": example_path / '6_BrightfieldNK92Data/example_output',
            "piv_parameters": {'window_size': 12.0, 'element_size': 4.8, 'signal_to_noise': 1.3, 'drift_correction': True},
            "mesh_parameters": {'reference_stack': 'next', 'element_size': 4.0, 'mesh_size': 'piv'},
            "material_parameters": {'k': 6062.0, 'd_0': 0.0025, 'lambda_s': 0.0804, 'ds':  0.034},
            "solve_parameters": {'alpha': 10**11, 'step_size': 0.33, 'max_iterations': 300, 'rel_conv_crit': 0.01},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/6_BrightfieldNK92Data.zip",
            "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/6_BrightfieldNK92Data_evaluated.zip",
            "url_evaluated_file": ["2023_02_14_12_0920_stack.saenopy"],
        },
    }
