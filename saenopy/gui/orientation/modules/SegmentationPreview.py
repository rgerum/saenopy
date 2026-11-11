"""Transient GUI previews using the existing segmentation functions."""
import numpy as np
from qtpy import QtCore


def calculate_preview(result, parameters, orientation):
    from CompactionAnalyzer.CompactionFunctions import create_axis_mask, normalize, segment_cell
    from skimage.color import rgb2gray

    image = result.get_image(0)
    if image.ndim == 3:
        image = rgb2gray(image)
    if result.uses_mask:
        edge = orientation.get("edge", 40)
        if not isinstance(edge, (int, np.integer)) or edge < 0 or 2 * edge >= min(image.shape):
            raise ValueError("Edge crop must leave a nonempty mask image.")
        crop = slice(edge, -edge) if edge else slice(None)
        mask = create_axis_mask(image[crop, crop], parameters["mask_sigma"],
                                parameters["mask_threshold"], parameters["mask_invert"])
        return {"mask": mask, "edge": edge}
    segmentation = segment_cell(normalize(image, 1, 99), thres=parameters["thresh"],
        seg_gaus1=parameters["gauss1"], seg_gaus2=parameters["gauss2"],
        show_segmentation=False, seg_invert=parameters["invert"], seg_iter=1,
        segmention_method="otsu", regional_max_correction=True, segmention_min_area=1000)
    return {"segmentation": segmentation}


class PreviewSignals(QtCore.QObject):
    finished = QtCore.Signal(int, object, object)


class PreviewWorker(QtCore.QRunnable):
    def __init__(self, revision, result, parameters, orientation):
        super().__init__()
        self.signals = PreviewSignals()
        self.arguments = revision, result, parameters, orientation

    def run(self):
        revision, result, parameters, orientation = self.arguments
        try:
            preview = calculate_preview(result, parameters, orientation)
        except Exception as error:
            self.signals.finished.emit(revision, None, str(error))
        else:
            self.signals.finished.emit(revision, preview, None)
