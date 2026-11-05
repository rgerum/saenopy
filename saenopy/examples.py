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
    for ex in [get_examples, get_examples_2D, get_examples_spheroid, get_examples_orientation]:
        examples = ex()
        if name in examples:
            example = examples[name]
            break
    else:
        raise KeyError
    url = example["url"]
    download_files(url, target_folder, progress_callback=progress_callback)

    if evaluated:
        evaluated_folder = Path(target_folder) / Path(Path(url).name).stem / "example_output"
        if example.get("evaluated_included", False):
            files = [evaluated_folder / file for file in example["url_evaluated_file"]]
            if not all(file.is_file() for file in files):
                raise FileNotFoundError(f"The example package is incomplete: missing evaluated results in {evaluated_folder}")
            return files
        if not (evaluated_folder / example["url_evaluated_file"][0]).exists():
            download_files(example["url_evaluated"], evaluated_folder, progress_callback=progress_callback)
        return [evaluated_folder / file for file in example["url_evaluated_file"]]


def get_examples():
    example_path = Path(appdirs.user_data_dir("saenopy", "rgerum"))
    image_path = Path(resource_path("thumbnails"))
    return {
        "ClassicSingleCellTFM": {
            "desc": "Hepatic stellate cells in 1.2mg/ml collagen with relaxed and deformed stacks.\nRelaxed state induced with cytochalasin D.\n3 examples (~800MB)",
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
            "url_evaluated_file": ["Pos004_S001_z{z}_ch{c00}_eval.saenopy", "Pos007_S001_z{z}_ch{c00}_eval.saenopy", "Pos008_S001_z{z}_ch{c00}_eval.saenopy"],
        },
        "DynamicalSingleCellTFM": {
            "desc": "Single natural killer cell that migrated through 1.2mg/ml collagen, recorded for 24min.\n1 example (~1.3GB)",
            "img": image_path / "Dynamic_icon.png",
            "voxel_size": [0.2407, 0.2407, 1.0071],
            "time_delta": 60,
            "stack": example_path / '2_DynamicalSingleCellTFM/data/Pos*_S001_t{t}_z{z}_ch{c:00}.tif',
            "output_path": example_path / '2_DynamicalSingleCellTFM/example_output',
            "piv_parameters": {'window_size': 12.0, 'element_size': 4.0, 'signal_to_noise': 1.3, 'drift_correction': True},
            "mesh_parameters": {'reference_stack': 'median', 'element_size': 4.0, 'mesh_size': 'piv'},
            "material_parameters": {'k': 1449.0, 'd_0': 0.0022, 'lambda_s': 0.032, 'd_s': 0.055},
            "solve_parameters": {'alpha': 10**10, 'step_size': 0.33, 'max_iterations': 100, 'rel_conv_crit': 0.01},
            "crop": {"z": (20, -20)},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/2_DynamicalSingleCellTFM.zip",
            "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/2_DynamicalSingleCellTFM_evaluated.zip",
            "url_evaluated_file": ["Pos002_S001_t{t}_z{z}_ch{c00}_eval.saenopy"],
        },
        "OrganoidTFM": {
            "desc": "Intestinal organoid in 1.2mg/ml collagen (~20MB)",
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
            "url_evaluated_file": ["Pos007_S001_t50_z{z}_ch00_eval.saenopy"],
        },
        "BrightfieldTFM": {
            "desc": "Traction forces around an immune cell in collagen 1.2mg/ml calculated on simple brightfield images (~1GB)",
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
            "url_evaluated_file": ["2023_02_14_12_0920_stack_eval.saenopy"],
        },
    }


def get_examples_2D():
    example_path = Path(appdirs.user_data_dir("saenopy", "rgerum"))
    image_path = Path(resource_path("thumbnails"))
    return {
        "WTKO": {
            "desc": "This data set contains raw data for 2 types of cell colonies: In one group a critical cytoskeletal protein has been knocked out.",
            "img": image_path / "pyTFM_WTKO.png",
            "pixel_size": 0.201,
            "bf": example_path / 'WTKO/*/*_bf_before.tif',
            "reference": example_path / 'WTKO/*/*_after.tif',
            "deformed": example_path / 'WTKO/*/[0-9][0-9]_before.tif',
            "output_path": example_path / 'WTKO/example_output',
            "piv_parameters": {'window_size': 100, 'overlap': 60, 'std_factor': 15},
            "force_parameters": {'young': 49000, 'sigma': 0.49, 'h': 300},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/WTKO.zip",
            "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/WTKO_evaluated.zip",
            "url_evaluated_file": ["KO/04_bf_before_eval.saenopy2D", "KO/05_bf_before_eval.saenopy2D", "WT/03_bf_before_eval.saenopy2D", "WT/10_bf_before_eval.saenopy2D"],
        },
    }


def get_examples_spheroid():
    example_path = Path(appdirs.user_data_dir("saenopy", "rgerum"))
    image_path = Path(resource_path("thumbnails"))
    return {
        "MCF7 spheroid": {
            "desc": "MCF7 spheroid contracting for 12h within a 1.2mg/ml collagen gel (~300MB)",
            "img": image_path / "MCF7-time-lapse.png",
            "pixel_size": 1.29,
            "time_delta": 300,
            "input": example_path / 'Jointforces_MCF7-time-lapse/20160912-122130_Mic3_rep?_pos*_x0_y0_modeBF_zMinProj.tif',
            "output_path": example_path / 'Jointforces_MCF7-time-lapse/example_output',
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Jointforces_MCF7-time-lapse.zip",
            "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Jointforces_MCF7-time-lapse_evaluated.zip",
            "url_evaluated_file": ["20160912-122130_Mic3_rep{t}_pos6_x0_y0_modeBF_zMinProj_eval.saenopySpheroid"],
        },
            "Human fibroblast spheroids": {
                "desc": "Human liver fibroblast spheroids (4000 cells) treated with and without inhibition of contractile forces using Rho-kinase inhibitor (added at 0h) for 10h in 1.2mg/ml collagen (~1GB)",
                "img": image_path / "SpheroidBlue.png",
                "pixel_size": 1.02385,
                "time_delta": 600,
                "input": example_path / 'Jointforces_FibroblastSpheroids_RockInhibition/*/*Mic5_rep?_pos*_in-focus_modeBF_slice0_z0.tif',
                "output_path": example_path / 'Jointforces_FibroblastSpheroids_RockInhibition/example_output',
                "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Jointforces_FibroblastSpheroids_RockInhibition.zip",
                "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Jointforces_FibroblastSpheroids_RockInhibition_evaluated.zip",
                "url_evaluated_file": ["Control/20201021-180200_Mic5_rep{t}_pos09_in-focus_modeBF_slice0_z0_eval.saenopySpheroid",
                                       "RockInhibitor10um/20201021-180200_Mic5_rep{t}_pos32_in-focus_modeBF_slice0_z0_eval.saenopySpheroid"],
                },
            "Mouse organoids": {
                "desc": "Mouse intestinal organoid responding to mechanosensitive agonist (added after 48h) for 3 days in 1.2mg/ml collagen (~1GB)",
                "img": image_path / "StainedOrganoid_icon.png",
                "pixel_size": 1.02385,
                "time_delta": 3600,
                "input": example_path / 'Jointforces_MouseOrganoids_MechanoAgonist/Mic5_rep?_pos*_in-focus_modeBF_slice0_z0.tiff',
                "output_path": example_path / 'Jointforces_MouseOrganoids_MechanoAgonist/example_output',
                "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Jointforces_MouseOrganoids_MechanoAgonist.zip",
                "url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Jointforces_MouseOrganoids_MechanoAgonist_evaluated.zip",
                "url_evaluated_file": ["Mic5_rep{t}_pos42_in-focus_modeBF_slice0_z0_eval.saenopySpheroid",
                                       "Mic5_rep{t}_pos43_in-focus_modeBF_slice0_z0_eval.saenopySpheroid",
                                       "Mic5_rep{t}_pos52_in-focus_modeBF_slice0_z0_eval.saenopySpheroid",
                                       "Mic5_rep{t}_pos129_in-focus_modeBF_slice0_z0_eval.saenopySpheroid"],
                }           
        }
       

def get_examples_orientation():
    example_path = Path(appdirs.user_data_dir("saenopy", "rgerum"))
    image_path = Path(resource_path("thumbnails"))
    fibroblast_url = "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Orientation_FibroblastActin.zip"
    collagen_url = "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Orientation_EmptyCollagen.zip"
    biological_url = "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Orientation_BiologicalExamples.zip"
    return {
        "CellFibers": {
            "title": "Cell-relative orientation",
            "desc": "Collagen alignment around a stained cell as a marker of contractility.",
            "img": image_path / "CellFibers.png",
            "pixel_size": 0.318,
            "input_cell": example_path / 'Orientation_CellsCollagen/data/Cell_*/C004.tif',
            "input_fiber": example_path / 'Orientation_CellsCollagen/data/Cell_*/C003.tif',
            "output_path": example_path / 'Orientation_CellsCollagen/eval',
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Orientation_CellsCollagen.zip",
            #"url_evaluated": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/MCF7-time-lapse_evaluated.zip",
            #"url_evaluated_file": ["KO/04_bf_before.saenopy2D", "KO/05_bf_before.saenopy2D", "WT/03_bf_before.saenopy2D", "WT/10_bf_before.saenopy2D"],
        },
        "FibroblastActin": {
            "title": "General orientation — same-image mask",
            "desc": "Actin fiber orientation within the cell area selected from the same phalloidin image.",
            "img": image_path / "Orientation_FibroblastActin.png",
            "pixel_size": 0.48,
            "input_fiber": example_path / "Orientation_FibroblastActin/data/field_*/phalloidin.tif",
            "input_cell": example_path / "Orientation_FibroblastActin/data/field_*/phalloidin.tif",
            "output_path": example_path / "Orientation_FibroblastActin/eval",
            "variants": [
                {"name": "phalloidin_mask", "analysis_mode": "axis_orientation_masked"},
            ],
            "orientation_parameters": {"sigma_tensor": 3., "sigma_tensor_type": "µm", "edge": 10,
                                       "sigma_first_blur": .5, "SaveNumpy": False, "plotting": False},
            "segmentation_parameters": {"mask_sigma": 5., "mask_threshold": .25, "mask_invert": False},
            "url": fibroblast_url,
            "url_evaluated": fibroblast_url,
            "evaluated_included": True,
            "url_evaluated_file": [f"phalloidin_mask/field_{i:02d}/phalloidin.saenopyOrientation" for i in (1, 2)],
        },
        "EmptyCollagen": {
            "title": "General orientation — whole image",
            "desc": "Collagen fiber orientation across an image without a cell reference or mask.",
            "img": image_path / "Orientation_EmptyCollagen.png",
            "pixel_size": 0.249,
            "input_fiber": example_path / "Orientation_EmptyCollagen/data/field_*/collagen.tif",
            "input_cell": None,
            "analysis_mode": "axis_orientation_only",
            "output_path": example_path / "Orientation_EmptyCollagen/eval",
            "orientation_parameters": {"sigma_tensor": 3., "sigma_tensor_type": "µm", "edge": 40,
                                       "sigma_first_blur": 1., "SaveNumpy": False, "plotting": False},
            "url": collagen_url,
            "url_evaluated": collagen_url,
            "evaluated_included": True,
            "url_evaluated_file": [f"field_{i:02d}/collagen.saenopyOrientation" for i in (1, 2)],
        },
        "ActininMask": {
            "title": "General orientation — second-image mask",
            "desc": "The alpha-actinin channel selects cardiomyocyte regions in a culture with fibroblasts. Measure the orientation of phalloidin-labeled actin within this second-image mask.",
            "img": image_path / "Orientation_ActininMask.png",
            "pixel_size": 1 / 6.411017,
            "input_fiber": example_path / "Orientation_BiologicalExamples/BiologicalExamples/cardiomyocyte_05/phalloidin.tif",
            "input_cell": example_path / "Orientation_BiologicalExamples/BiologicalExamples/cardiomyocyte_05/actinin.tif",
            "analysis_mode": "axis_orientation_masked",
            "output_path": example_path / "Orientation_BiologicalExamples/eval_cardiomyocyte",
            "orientation_parameters": {"sigma_tensor": 8., "sigma_tensor_type": "pixel", "edge": 15,
                                       "sigma_first_blur": .5, "SaveNumpy": False, "plotting": False},
            "segmentation_parameters": {"mask_sigma": 8., "mask_threshold": .65, "mask_invert": False},
            "url": biological_url, "url_evaluated": biological_url, "evaluated_included": True,
            "url_evaluated_file": ["cardiomyocyte/cardiomyocyte.saenopyOrientation"],
        },
        "SpheroidPoint": {
            "title": "Orientation towards a point — spheroid",
            "desc": "Orientation of peripheral cell structures towards the spheroid center. PNG-derived grayscale example; scale estimated from the embedded bar.",
            "img": image_path / "Orientation_SpheroidPoint.png",
            "pixel_size": 1.,
            "input_fiber": example_path / "Orientation_BiologicalExamples/BiologicalExamples/spheroid/orientation.tif",
            "input_cell": example_path / "Orientation_BiologicalExamples/BiologicalExamples/spheroid/selection.tif",
            "analysis_mode": "point_orientation",
            "output_path": example_path / "Orientation_BiologicalExamples/eval_spheroid",
            "orientation_parameters": {"sigma_tensor": 6., "sigma_tensor_type": "pixel", "edge": 15,
                                       "reference_x": 1200., "reference_y": 910.,
                                       "sigma_first_blur": .5, "SaveNumpy": False, "plotting": False},
            "segmentation_parameters": {"mask_sigma": 0., "mask_threshold": .5, "mask_invert": False},
            "url": biological_url, "url_evaluated": biological_url, "evaluated_included": True,
            "url_evaluated_file": ["spheroid/spheroid.saenopyOrientation"],
        },
        "PointReference": {
            "title": "Orientation towards a point",
            "desc": "Choose a point in a collagen image and measure radial fiber alignment towards it.",
            "img": image_path / "Orientation_PointReference.png",
            "pixel_size": 0.318,
            "input_fiber": example_path / "Orientation_CellsCollagen/data/Cell_1/C003.tif",
            "input_cell": None,
            "analysis_mode": "point_orientation",
            "output_path": example_path / "Orientation_CellsCollagen/eval_point",
            "orientation_parameters": {"sigma_tensor": 7., "sigma_tensor_type": "µm", "edge": 40,
                                       "sigma_first_blur": .5, "SaveNumpy": False, "plotting": False},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Orientation_CellsCollagen.zip",
        },
        "CardiomyocyteLine": {
            "title": "Orientation relative to a line — cardiomyocytes",
            "desc": "Click two points along the cardiomyocytes to define a reference line. Analyze phalloidin orientation throughout the image, without a mask.",
            "img": image_path / "Orientation_CardiomyocyteLine.png",
            "pixel_size": 1 / 6.411017,
            "input_fiber": example_path / "Orientation_BiologicalExamples/BiologicalExamples/cardiomyocyte_line/orientation.tif",
            "input_cell": None,
            "analysis_mode": "line_orientation",
            "output_path": example_path / "Orientation_BiologicalExamples/eval_cardiomyocyte_line",
            "orientation_parameters": {"sigma_tensor": 8., "sigma_tensor_type": "pixel", "edge": 15,
                                       "line_x1": 69., "line_y1": 469., "line_x2": 655., "line_y2": 255.,
                                       "sigma_first_blur": .5, "SaveNumpy": False, "plotting": False},
            "segmentation_parameters": {},
            "url": biological_url, "url_evaluated": biological_url, "evaluated_included": True,
            "url_evaluated_file": ["cardiomyocyte_line/cardiomyocyte_line.saenopyOrientation"],
        },
        "LineReference": {
            "title": "Orientation relative to a line",
            "desc": "Click two points in a collagen image and measure fiber alignment relative to that line.",
            "img": image_path / "Orientation_LineReference.png",
            "pixel_size": 0.318,
            "input_fiber": example_path / "Orientation_CellsCollagen/data/Cell_1/C003.tif",
            "input_cell": None,
            "analysis_mode": "line_orientation",
            "output_path": example_path / "Orientation_CellsCollagen/eval_line",
            "orientation_parameters": {"sigma_tensor": 7., "sigma_tensor_type": "µm", "edge": 40,
                                       "sigma_first_blur": .5, "SaveNumpy": False, "plotting": False},
            "url": "https://github.com/rgerum/saenopy/releases/download/v0.7.4/Orientation_CellsCollagen.zip",
        },
    }
