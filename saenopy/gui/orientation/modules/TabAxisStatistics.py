from qtpy import QtWidgets
from saenopy.gui.common import QtShortCuts
from saenopy.gui.common.TabModule import TabModule
from saenopy.gui.common.gui_classes import MatplotlibWidget


class TabAxisStatistics(TabModule):
    """Axis statistics from the same valid pixels as the stored maps."""
    def __init__(self, parent):
        super().__init__(parent)
        with self.parent.tabs.createTab("Axis histogram") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.summary = QtWidgets.QLabel().addToLayout()
                self.summary.setWordWrap(True)
                self.canvas = MatplotlibWidget(self)
                layout.addWidget(self.canvas)

    def checkTabEnabled(self, result):
        return result is not None and result.axis_histogram is not None

    def update_display(self, **kwargs):
        if not self.checkTabEnabled(self.result):
            return
        summary = self.result.results_total[0]
        histogram = self.result.axis_histogram
        self.summary.setText(f"{summary['Valid pixels']} valid pixels. Mean alignment: "
                             f"x = {summary['Alignment x']:.3f}, y = {summary['Alignment y']:.3f}. "
                             f"Mean absolute deviation: x = {45*(1-summary['Alignment x']):.1f}°, "
                             f"y = {45*(1-summary['Alignment y']):.1f}°. "
                             "Each pixel has equal weight. +1 parallel, -1 perpendicular.")
        self.canvas.figure.clear()
        axes = self.canvas.figure.subplots(1, 2)
        for index, axis in enumerate(("x", "y")):
            counts = histogram[:, index + 2]
            fraction = counts / counts.sum() if counts.sum() else counts
            axes[index].bar(histogram[:, 0], fraction, width=histogram[:, 1]-histogram[:, 0],
                            align="edge", color="steelblue", edgecolor="white")
            axes[index].set(xlim=(-90, 90), xlabel=f"Angle towards {axis}-axis (deg)", ylabel="Fraction of valid pixels")
        self.canvas.figure.tight_layout()
        self.canvas.draw()
