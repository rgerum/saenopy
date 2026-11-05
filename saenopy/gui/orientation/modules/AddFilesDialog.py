# Setting the Qt bindings for QtPy
import os
import glob

from qtpy import QtCore, QtWidgets, QtGui
from saenopy.gui.common import QtShortCuts
from saenopy.examples import get_examples_orientation
from saenopy.gui.common.AddFilesDialog import AddFilesDialog
from .AnalysisModeSelector import AnalysisModeSelector
from .MaskSourceSelector import MaskSourceSelector
from .AxisSelector import AxisSelector


class AddFilesDialog(AddFilesDialog):
    settings_group = "open_orientation"
    file_extension = ".saenopyOrientation"

    examples_list = get_examples_orientation()

    def __init__(self, parent, settings):
        self.initial_analysis_mode = getattr(parent, "analysis_mode", "full")
        self.initial_mask_source = getattr(parent, "axis_mask_source",
            "separate_image" if self.initial_analysis_mode == "axis_orientation_masked" else "whole_image")
        super().__init__(parent, settings)

    def selected_analysis_mode(self):
        if self.analysis_mode_selector.value() == "full":
            return "full"
        if self.analysis_mode_selector.value() == "point":
            return self.reference_type.value() + "_orientation"
        return "axis_orientation_only" if self.mask_source.value() == "whole_image" else "axis_orientation_masked"

    def selected_cell_pattern(self):
        if self.selected_analysis_mode() == "full":
            return self.cellText.value()
        if self.mask_source.value() == "whole_image":
            return None
        if self.mask_source.value() == "same_image":
            return self.fiberText.value()
        return self.cellText.value()

    def add_new_measurement_tab(self):
        with self.tabs.createTab("New Measurement") as self.tab:
            with QtShortCuts.QVBoxLayout() as layout:
                self.analysis_mode_selector = AnalysisModeSelector(self.initial_analysis_mode)
                layout.addWidget(self.analysis_mode_selector)
                self.reference_type = AxisSelector(
                    "line" if self.initial_analysis_mode == "line_orientation" else "point",
                    choices=[("point", "Point"), ("line", "Line")], title="Reference")
                layout.addWidget(self.reference_type)
                self.mask_source = MaskSourceSelector(self.initial_mask_source)
                layout.addWidget(self.mask_source)
                self.label = QtWidgets.QLabel()
                self.label.setWordWrap(True)
                layout.addWidget(self.label)

                self.cellText = QtShortCuts.QInputFilename(None, "Cell Images", file_type="Image (*.tif *.png *.jpg)",
                                                           settings=self.settings,
                                                           settings_key=f"{self.settings_group}/wildcard_cell", existing=True,
                                                           allow_edit=True)
                self.fiberText = QtShortCuts.QInputFilename(None, "Orientation Images", file_type="Image (*.tif *.png *.jpg)",
                                                            settings=self.settings,
                                                            settings_key=f"{self.settings_group}/wildcard_fiber", existing=True,
                                                            allow_edit=True)
                self.outputText = QtShortCuts.QInputFolder(None, "output", settings=self.settings,
                                                           settings_key=f"{self.settings_group}/output_folder", allow_edit=True)
                
                self.pixel_size = QtShortCuts.QInputString(None, "pixel size", 1.29, settings=self.settings,
                                                           settings_key=f"{self.settings_group}/pixel_size", unit="µm",
                                                           allow_none=False, type=float)
                
                
                

                def changed(*args):
                    mode = self.selected_analysis_mode()
                    source = self.mask_source.value()
                    self.mask_source.setVisible(mode != "full")
                    self.reference_type.setVisible(mode in ("point_orientation", "line_orientation"))
                    whole_image = mode == "axis_orientation_only"
                    needs_second_image = mode == "full" or (mode != "full" and source == "separate_image")
                    self.cellText.setEnabled(needs_second_image)
                    self.cellText.setVisible(needs_second_image)
                    self.cellText.setLabel("Cell Images" if mode == "full" else "Mask Images")
                    if mode == "full":
                        self.label.setText("Select orientation images and cell images to measure alignment towards the cell as a proxy for contractility. "
                                           "Use * for batches; the two lists are paired in sorted filename order.")
                    elif mode == "point_orientation":
                        self.label.setText("Select orientation images. Choose a reference point in each image, then measure radial alignment towards that point.")
                    elif mode == "line_orientation":
                        self.label.setText("Select orientation images. Click two points in each image to define a line, then measure alignment parallel to that line in the selected area.")
                    elif whole_image:
                        self.label.setText("Select orientation images to measure orientation towards the x and y axes over the whole image.")
                    elif source == "same_image":
                        self.label.setText("Select orientation images. A mask is created from each image; "
                                           "orientation is measured inside it. Adjust mask blur and threshold using the preview.")
                    else:
                        self.label.setText("Select orientation images and matching mask images. Orientation is calculated from the orientation image; "
                                           "the mask image only selects the area. The lists are paired in sorted filename order.")
                    if mode in ("point_orientation", "line_orientation") and source != "whole_image":
                        self.label.setText(self.label.text() + " The mask selects the area; it does not change the reference.")
                    fiber = glob.glob(self.fiberText.value()) if self.fiberText.value() else []
                    cells = glob.glob(self.cellText.value()) if self.cellText.value() else []
                    valid = bool(fiber) and (not needs_second_image or len(fiber) == len(cells)) and bool(self.outputText.value())
                    region = "whole image" if whole_image else ("mask from each orientation image" if mode != "full" and source == "same_image" else f"{len(cells)} cell/mask images")
                    if mode == "point_orientation":
                        region = "one reference point per image"
                    elif mode == "line_orientation":
                        region = "one reference line per image (two clicks)"
                    self.label2.setText(f"{len(fiber)} orientation images; {region}")
                    self.label2.setStyleSheet("QLabel { color: " + ("green" if valid else "red") + "; }")
                    self.button_addList1.setEnabled(valid)

                self.analysis_mode_selector.mode_selected.connect(changed)
                self.reference_type.valueChanged.connect(changed)
                self.mask_source.valueChanged.connect(changed)
                self.outputText.valueChanged.connect(changed)
                self.fiberText.line.textChanged.connect(changed)
                self.cellText.line.textChanged.connect(changed)
                self.label2 = QtWidgets.QLabel()  # .addToLayout()
                layout.addWidget(self.label2)
                layout.addStretch()

                with QtShortCuts.QHBoxLayout() as layout3:
                    # self.button_clear = QtShortCuts.QPushButton(None, "clear list", self.clear_files)
                    layout3.addStretch()
                    self.button_addList2 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                    self.button_addList1 = QtShortCuts.QPushButton(None, "ok", self.accept_new)
                changed()

    def add_examples_tab(self):
        # Match the README: cell first, general orientation, then point/line.
        with self.tabs.createTab("Examples") as self.tab4:
            self.example_buttons = []
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.tab4.addWidget(scroll)
            content = QtWidgets.QWidget()
            scroll.setWidget(content)
            sections = QtWidgets.QVBoxLayout(content)
            sections.setContentsMargins(0, 0, 0, 0)
            sections.setSpacing(14)
            for category, title, subtitle, accent, background, names in [
                ("cell", "Cell-relative orientation", "Marker of cell contractility",
                 "#2563eb", "#eff6ff", ["CellFibers"]),
                ("axis", "General orientation", "Orientation relative to X and Y; whole image or optional mask",
                 "#0f766e", "#f0fdfa", ["EmptyCollagen", "FibroblastActin", "ActininMask"]),
                ("reference", "Orientation towards a point or relative to a line", "Choose the reference in the image",
                 "#7c3aed", "#f5f3ff", ["PointReference", "SpheroidPoint", "LineReference", "CardiomyocyteLine"]),
            ]:
                section = QtWidgets.QFrame()
                section.setObjectName(f"exampleCategory_{category}")
                section.setStyleSheet(f"""
                    QFrame#exampleCategory_{category} {{
                        background: {background}; border: 1px solid {accent};
                        border-left: 5px solid {accent}; border-radius: 7px;
                    }}
                    QLabel {{ color: #1f2937; background: transparent; border: none; }}
                    QLabel#categoryHeading {{ color: {accent}; font-size: 16px; font-weight: 600; }}
                    QFrame#exampleCard {{ background: white; border: 1px solid #d1d5db; border-radius: 5px; }}
                    QLabel#exampleTitle {{ color: {accent}; font-weight: 600; }}
                    QLabel#examplePlaceholder {{ color: #64748b; background: #f8fafc; border: 1px dashed #cbd5e1; }}
                    QPushButton {{ color: {accent}; background: white; border: 1px solid {accent};
                                   border-radius: 4px; padding: 5px 8px; }}
                    QPushButton:hover {{ background: {background}; }}
                    QPushButton:focus {{ border: 2px solid {accent}; }}
                    QPushButton:disabled {{ color: #64748b; border-color: #cbd5e1; background: #f8fafc; }}
                """)
                section_layout = QtWidgets.QVBoxLayout(section)
                heading = QtWidgets.QLabel(title)
                heading.setObjectName("categoryHeading")
                section_layout.addWidget(heading)
                section_layout.addWidget(QtWidgets.QLabel(subtitle))
                cards = QtWidgets.QGridLayout() if category == "reference" else QtWidgets.QHBoxLayout()
                cards.setSpacing(10)
                for index, name in enumerate(names):
                    card = self._example_card(name, compact=category == "cell")
                    if category == "reference":
                        cards.addWidget(card, index // 2, index % 2)
                    else:
                        cards.addWidget(card, 1)
                section_layout.addLayout(cards)
                sections.addWidget(section)
            sections.addStretch()
            self.download_state = QtWidgets.QLabel("")
            self.tab4.addWidget(self.download_state)
            self.download_progress = QtWidgets.QProgressBar()
            self.download_progress.setRange(0, 100)
            self.tab4.addWidget(self.download_progress)

    def _example_card(self, name, compact=False):
        properties = self.examples_list.get(name)
        card = QtWidgets.QFrame()
        card.setObjectName("exampleCard")
        card.setProperty("exampleName", name)
        card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        outer = QtWidgets.QHBoxLayout(card) if compact else QtWidgets.QVBoxLayout(card)
        layout = QtWidgets.QVBoxLayout() if compact else outer
        if compact:
            outer.addLayout(layout, 1)
        title = QtWidgets.QLabel(properties.get("title", name) if properties else "General orientation — second-image mask")
        title.setObjectName("exampleTitle")
        title.setWordWrap(True)
        if not compact:
            title.setMinimumHeight(2 * title.fontMetrics().height())
        layout.addWidget(title)
        description = QtWidgets.QLabel(properties["desc"] if properties else
            "In a mixed culture of cardiomyocytes and fibroblasts, a separate alpha-actinin stain selects "
            "the cardiomyocyte regions. Fiber orientation is evaluated only within this mask.")
        description.setWordWrap(True)
        thumbnail = QtWidgets.QLabel()
        thumbnail.setAlignment(QtCore.Qt.AlignCenter)
        thumbnail.setFixedHeight(92)
        if properties:
            pixmap = QtGui.QPixmap(str(properties["img"]))
            thumbnail.setPixmap(pixmap.scaled(145, 92, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        else:
            thumbnail.setText("Image pair not yet available")
            thumbnail.setObjectName("examplePlaceholder")
            thumbnail.setWordWrap(True)
        if compact:
            layout.addWidget(description)
            outer.addWidget(thumbnail)
        else:
            layout.addWidget(thumbnail)
            layout.addWidget(description)
        layout.addStretch()
        buttons = QtWidgets.QHBoxLayout()
        if properties:
            for evaluated, text in [(False, "Open"), (True, "Open (evaluated)")]:
                button = QtWidgets.QPushButton(text)
                button.setEnabled(not evaluated or properties.get("url_evaluated") is not None)
                button.clicked.connect(lambda checked=False, name=name, evaluated=evaluated: self.load_example(name, evaluated))
                buttons.addWidget(button)
                self.example_buttons.append(button)
        else:
            button = QtWidgets.QPushButton("Example pending")
            button.setEnabled(False)
            buttons.addWidget(button)
        if compact:
            buttons.addStretch()
        layout.addLayout(buttons)
        return card
