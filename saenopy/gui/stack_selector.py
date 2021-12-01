import os
import sys
import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from saenopy.gui import QtShortCuts
from saenopy.gui.stack_selector_leica import StackSelectorLeica
from saenopy.gui.stack_selector_tif import StackSelectorTif
from saenopy.gui import QtShortCuts, QExtendedGraphicsView
from qimage2ndarray import array2qimage
from datetime import timedelta
import datetime
import tifffile
import imageio
from pathlib import Path
from typing import Union, Optional, Any
from numpy import float64, int32, ndarray

def timedelta_mul(self: timedelta, other: float) -> timedelta:
    if isinstance(other, (int, float)):
        return datetime.timedelta(seconds=self.total_seconds() * other)
    else:
        return NotImplemented


def timedelta_div(self: timedelta, other: int) -> timedelta:
    if isinstance(other, (int, float)):
        return datetime.timedelta(seconds=self.total_seconds() / other)
    else:
        return NotImplemented


def BoundBy(value: Any, min: Any, max: Any) -> Any:
    if value is None:
        return min
    if value < min:
        return min
    if value > max:
        return max
    return value


def roundTime(dt: Optional[datetime.datetime] = None, roundTo: float = 60) -> datetime:
    """Round a datetime object to any time laps in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt == None: dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def roundValue(value, modulo, offset=0):
    return int((value - offset) // modulo) * modulo + offset


def DateDivision(x, y):
    return x.total_seconds() / y.total_seconds()


def Remap(value: Any, minmax1: list, minmax2: list) -> Any:
    length1 = minmax1[1] - minmax1[0]
    length2 = minmax2[1] - minmax2[0]
    if length1 == 0:
        return 0
    try:
        percentage = (value - minmax1[0]) / length1
    except TypeError:
        percentage = DateDivision((value - minmax1[0]), length1)
    try:
        value2 = percentage * length2 + minmax2[0]
    except TypeError:
        value2 = datetime.timedelta(seconds=percentage * length2.total_seconds()) + minmax2[0]
    return value2

class TimeLineGrabber(QtWidgets.QGraphicsPathItem):
    def __init__(self, parent: Union["TimeLineSlider", "RealTimeSlider"], value: int, path: QtGui.QPainterPath,
                 gradient: QtGui.QLinearGradient, parent_item: Optional[QtWidgets.QGraphicsPathItem] = None) -> None:
        if parent_item is None:
            QtWidgets.QGraphicsPathItem.__init__(self, parent.parent)
        else:
            QtWidgets.QGraphicsPathItem.__init__(self, parent_item)
        self.parent = parent
        self.pixel_range = [0, 100]
        self.value_range = [0, 100]
        self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.dragged = False
        self.draggedY = False

        self.setPath(path)
        self.setBrush(QtGui.QBrush(gradient))
        self.setZValue(10)
        self.value = value

        self.signal = TimeLineGrabberSignal()

    def setPixelRange(self, min: float, max: float) -> None:
        self.pixel_range = [min, max]
        self.updatePos()

    def setValueRange(self, min: Any, max: Any) -> None:
        self.value_range = [min, max]

    def setValue(self, value: Any) -> None:
        self.value = int(round(value))
        self.updatePos()

    def updatePos(self) -> None:
        if 0:
            self.setPos(self.value_to_pixel(self.value), 0)
        else:
            self.setPos(0, self.value_to_pixel(self.value))

    def mousePressEvent(self, event: QtCore.QEvent) -> None:
        if event.button() == 1:
            self.dragged = True
            self.signal.sliderPressed.emit()
        if event.button() == 2:
            self.draggedY = True

    def mouseMoveEvent(self, event: QtCore.QEvent) -> None:
        if self.dragged:
            if 0:
                x = BoundBy(self.mapToParent(event.pos()).x(), self.pixel_range[0], self.pixel_range[1])
                self.setValue(self.pixel_to_value(x))
                self.signal.sliderMoved.emit()
            else:
                y = BoundBy(self.mapToParent(event.pos()).y(), self.pixel_range[0], self.pixel_range[1])
                self.setValue(self.pixel_to_value(y))
                self.signal.sliderMoved.emit()
        if self.draggedY:
            y = BoundBy(self.mapToParent(event.pos()).y(), self.pixel_range[0], self.pixel_range[1])
            delta = np.abs(self.value-self.pixel_to_value(y))
            self.parent.setActiveRange(delta)

    def mouseReleaseEvent(self, event: QtCore.QEvent) -> None:
        self.dragged = False
        self.draggedY = False
        self.signal.sliderReleased.emit()

    def pixel_to_value(self, pixel: Any) -> Any:
        return Remap(pixel, self.pixel_range, self.value_range)

    def value_to_pixel(self, value: Any) -> Any:
        return Remap(value, self.value_range, self.pixel_range)


class TimeLineGrabberTime(TimeLineGrabber):
    # def __init__(self, *args):
    #    QGraphicsPathItem.__init__(self, None, parent.scene)

    def mouseMoveEvent(self, event: QtCore.QEvent) -> None:
        if self.dragged:
            x = self.pos().x() + event.pos().x() / self.parent.scale / self.parent.slider_line.transform().m11()
            self.setValue(self.pixel_to_value(x))

    def setValue(self, value: datetime) -> None:
        self.value = BoundBy(value, *self.value_range)
        self.updatePos()


class TimeLineGrabberSignal(QtCore.QObject):
    sliderPressed = QtCore.Signal()
    sliderMoved = QtCore.Signal()
    sliderReleased = QtCore.Signal()

class TimeLineSlider(QtWidgets.QGraphicsView):
    start_changed = QtCore.Signal(int)
    end_changed = QtCore.Signal(int)
    valueChanged = QtCore.Signal(int)
    activeRangeChanged = QtCore.Signal(int)

    def __init__(self, max_value: int = 0, min_value: int = 0, scale: float = 1) -> None:
        QtWidgets.QGraphicsView.__init__(self)

        self.setMaximumWidth(15 * scale)
        if scale != 1:
            self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, self.sizePolicy().verticalPolicy())

        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.parent = QtWidgets.QGraphicsRectItem(None)
        self.parent.setScale(scale)
        self.scene.addItem(self.parent)
        self.scene.setBackgroundBrush(self.palette().color(QtGui.QPalette.Background))
        self.setStyleSheet("border: 0px")

        self.max_value = max_value
        self.min_value = min_value

        self.slider_line = QtWidgets.QGraphicsRectItem(self.parent)
        self.slider_line.setPen(QtGui.QPen(QtGui.QColor("black")))
        self.slider_line.setPos(0, -2.5)
        gradient = QtGui.QLinearGradient(QtCore.QPointF(0, 0), QtCore.QPointF(5, 0))
        gradient.setColorAt(0, QtGui.QColor("black"))
        gradient.setColorAt(1, QtGui.QColor(128, 128, 128))
        self.slider_line.setBrush(QtGui.QBrush(gradient))
        self.slider_line.mousePressEvent = self.SliderBarMousePressEvent

        self.slider_line_active = QtWidgets.QGraphicsRectItem(self.parent)
        self.slider_line_active.setPen(QtGui.QPen(QtGui.QColor("black")))
        self.slider_line_active.setPos(0, -2.5)
        gradient = QtGui.QLinearGradient(QtCore.QPointF(0, 0), QtCore.QPointF(5, 0))
        gradient.setColorAt(0, QtGui.QColor(128, 128, 128))
        gradient.setColorAt(1, QtGui.QColor(200, 200, 200))
        self.slider_line_active.setBrush(QtGui.QBrush(gradient))

        path = QtGui.QPainterPath()
        path.addRect(-4, -5, 14, 5)
        gradient = QtGui.QLinearGradient(QtCore.QPointF(-7, 0), QtCore.QPointF(14, 0))
        gradient.setColorAt(0, QtGui.QColor(255, 0, 0))
        gradient.setColorAt(1, QtGui.QColor(128, 0, 0))
        self.slider_position = TimeLineGrabber(self, 0, path, gradient)
        self.slider_position.signal.sliderMoved.connect(self.updatePlayRange)
        self.slider_position.signal.sliderMoved.connect(lambda: self.valueChanged.emit(self.value()))

        self.length = 1
        self.active_range = 0

        self.tick_marker = {}

        self.slider_position.signal.sliderReleased.connect(lambda : self.valueChanged.emit(self.value()))

    def SliderBarMousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        self.setValue(self.PixelToValue(self.slider_line.mapToScene(event.pos()).y()))
        self.slider_position.signal.sliderReleased.emit()
        self.updatePlayRange()

    def setActiveRange(self, value):
        self.active_range = value
        self.activeRangeChanged.emit(self.active_range)
        self.updatePlayRange()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if 1:
            self.length = (self.size().height() - 6) / self.parent.scale()
            self.slider_line.setRect(0, 0, 5, self.length)
            if 0:
                self.slider_line_active.setRect(0, self.ValueToPixel(self.slider_start.value),
                                                5,
                                                self.ValueToPixel(self.slider_end.value) - self.ValueToPixel(
                                                    self.slider_start.value))
            self.ensureVisible(self.slider_line)
            for marker in [self.slider_position]:
                marker.setPixelRange(0, self.length)
            self.repaint()

            return
        self.length = (self.size().width() - 20) / self.parent.scale()
        self.slider_line.setRect(0, 0, self.length, 5)
        self.slider_line_active.setRect(self.ValueToPixel(self.slider_start.value), 0,
                                        self.ValueToPixel(self.slider_end.value) - self.ValueToPixel(
                                            self.slider_start.value), 5)
        self.ensureVisible(self.slider_line)
        for pos, ticks in self.tick_marker.items():
            for type, tick in ticks.items():
                tick.setPos(self.ValueToPixel(pos), 0)
                width = self.ValueToPixel(1)
                if pos == self.max_value:
                    width = 2
                tick.setRect(0.0, -3.5, width, -tick.height)
        for marker in [self.slider_position, self.slider_start, self.slider_end]:
            marker.setPixelRange(0, self.length)
        self.repaint()

    def setRange(self, min_value: int, max_value: int) -> None:
        self.min_value = min_value
        self.max_value = max_value
        for marker in [self.slider_position]:#, self.slider_start, self.slider_end]:
            marker.setValueRange(self.min_value, self.max_value)

    def setValue(self, value: float) -> None:
        self.slider_position.setValue(BoundBy(value, self.min_value, self.max_value))
        self.updatePlayRange()

    def PixelToValue(self, pixel: float) -> float:
        return Remap(pixel, [0, self.length], [self.min_value, self.max_value])

    def ValueToPixel(self, value: Union[int32, int]) -> Union[float, float64, int]:
        return Remap(value, [self.min_value, self.max_value], [0, self.length])

    def updatePlayRange(self) -> None:
        if 0:
            start = max([self.slider_position.value-self.active_range, self.min_value])
            end = min([self.slider_position.value+self.active_range, self.max_value])
            self.slider_line_active.setRect(self.ValueToPixel(start), 0,
                                            self.ValueToPixel(end) - self.ValueToPixel(start), 5)
        else:
            start = max([self.slider_position.value - self.active_range, self.min_value])
            end = min([self.slider_position.value + self.active_range, self.max_value])
            self.slider_line_active.setRect(0, self.ValueToPixel(start), 5,
                                            self.ValueToPixel(end) - self.ValueToPixel(start))

    def value(self) -> int:
        return self.slider_position.value

    def activeRange(self) -> int:
        return self.active_range

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        event.setAccepted(False)
        return


class ModuleScaleBar(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self)
        self.parent = parent

        self.font = QtGui.QFont()
        self.font.setPointSize(16)

        self.scale = 1

        self.scalebar = QtWidgets.QGraphicsRectItem(0, 0, 1, 1, self.parent.view.hud_lowerRight)
        self.scalebar.setBrush(QtGui.QBrush(QtGui.QColor("white")))
        self.scalebar.setPen(QtGui.QPen(QtGui.QColor("white")))
        self.scalebar.setPos(-20, -20)
        self.scalebar_text = QtWidgets.QGraphicsTextItem("", self.parent.view.hud_lowerRight)
        self.scalebar_text.setFont(self.font)
        self.scalebar_text.setDefaultTextColor(QtGui.QColor("white"))

        self.parent.signal_zoom.connect(self.zoomEvent)
        #self.parent.view.zoomEvent = lambda scale, pos: self.zoomEvent(scale, pos)
        #self.parent.signal_objective_changed.connect(self.updateStatus)
        #self.parent.signal_coupler_changed.connect(self.updateStatus)

        self.updateStatus()

    def updateStatus(self):
        self.updateBar()

    def zoomEvent(self, scale, pos):
        self.scale = scale
        self.updateBar()

    def updateBar(self):
        if self.parent.getVoxelSize() is None:
            return
        self.pixtomu = self.parent.getVoxelSize()[0]
        if self.scale == 0:
            return
        mu = 100*self.pixtomu/self.scale
        values = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 5000, 10000]
        old_v = mu
        for v in values:
            if mu < v:
                mu = old_v
                break
            old_v = v
        pixel = mu/(self.pixtomu)*self.scale
        self.scalebar.setRect(0, 0, -pixel, 5)
        self.scalebar_text.setPos(-pixel-20-25, -20-30)
        self.scalebar_text.setTextWidth(pixel+50)
        self.scalebar_text.setHtml(u"<center>%d&thinsp;µm</center>" % mu)


class StackSelector(QtWidgets.QWidget):
    active = None
    signal_zoom = QtCore.Signal(object, object)
    stack_changed = QtCore.Signal()
    glob_string_changed = QtCore.Signal(str, object)

    def __init__(self, layout, name, partner=None, use_time=False):
        super().__init__()
        self.name = name
        self.partner = partner

        self.settings = QtCore.QSettings("Saenopy", "Seanopy")


        main_layout = QtWidgets.QVBoxLayout(self)

        self._open_dir = self.settings.value("_open_dir" + self.name)
        if self._open_dir is None:
            self._open_dir = ""

        self.input_filename = QtShortCuts.QInputFilename(main_layout, name, self._open_dir, existing=True,
                                                         settings=self.settings, settings_key=self.name,
                                                         file_type="images (*.tif *.jpg *.png *.lif)")

        self.input_filename.valueChanged.connect(self.file_changed)

        self.selectors = []
        for selector in [StackSelectorLeica, StackSelectorTif]:
            selector_instance = selector(self, use_time=use_time)
            self.selectors.append(selector_instance)
            main_layout.addWidget(selector_instance)

        layout_image = QtWidgets.QHBoxLayout()
        main_layout.addLayout(layout_image)
        self.view = QExtendedGraphicsView.QExtendedGraphicsView(self)
        layout_image.addWidget(self.view)
        self.pixmap = QtWidgets.QGraphicsPixmapItem(self.view.origin)
        self.z_slider = TimeLineSlider()#QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.z_slider.setEnabled(False)
        self.z_slider.valueChanged.connect(self.zValueChanged)
        self.z_slider.activeRangeChanged.connect(lambda x: self.zValueChanged(self.z_slider.value()))

        layout_image.addWidget(self.z_slider)

        if self.partner is not None:
            layout_link = QtWidgets.QHBoxLayout()
            layout_link.setContentsMargins(0, 0, 0, 0)
            main_layout.addLayout(layout_link)

            self.input_linked = QtShortCuts.QInputBool(layout_link, "Link views", False)
            self.input_linked.valueChanged.connect(self.setLinked)
            self.partner.z_slider.valueChanged.connect(self.partner_zValueChanged)

            self.button_export = QtWidgets.QPushButton("export")
            layout_link.addWidget(self.button_export)
            self.button_export.clicked.connect(self.export)
        else:
            self.bottom_label = QtWidgets.QLabel()
            main_layout.addWidget(self.bottom_label)

        self.scalebar = ModuleScaleBar(self)

        def zoomEvent(scale, pos):
            self.signal_zoom.emit(scale, pos)

        self.view.zoomEvent = zoomEvent
        self.view.panEvent = zoomEvent

        layout.addWidget(self)

    def export(self):
        new_path = QtWidgets.QFileDialog.getSaveFileName(None, "Save Images", os.getcwd())
        # if we got one, set it
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
            new_path = new_path.strip(".gif").strip("_relaxed.tif").strip("_deformed.tif")
            new_path = Path(new_path)
            tifffile.imsave(new_path.parent / (new_path.stem + "_relaxed.tif"), self.im)
            tifffile.imsave(new_path.parent / (new_path.stem + "_deformed.tif"), self.partner.im)
            imageio.mimsave(new_path.parent / (new_path.stem + ".gif"), [self.partner.im, self.im], fps=2)

    def setImage(self, im):
        self.im = im
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view.setExtend(im.shape[1], im.shape[0])

    def file_changed(self, filename):
        print("file_changed", filename)
        for selector_instance in self.selectors:
            selector_instance.setVisible(False)

        for selector_instance in self.selectors:
            if selector_instance.checkAcceptFilename(filename):
                selector_instance.setImage(filename)
                self.active = selector_instance
                self.z_slider.setEnabled(True)

    def setLinked(self, value):
        if value is True:
            #self.z_slider.setEnabled(False)
            self.partner.z_slider.setValue(self.z_slider.value())
            def changes1(*args):
                self.view.setOriginScale(self.partner.view.getOriginScale())
                start_x, start_y, end_x, end_y = self.partner.view.GetExtend()
                self.view.centerOn(start_x+(end_x-start_x)/2, start_y+(end_y-start_y)/2)
            def zoomEvent(scale, pos):
                changes1()
                self.signal_zoom.emit(scale, pos)
                self.partner.signal_zoom.emit(scale, pos)
            self.partner.view.zoomEvent = zoomEvent
            self.partner.view.panEvent = changes1

            def changes2(*args):
                self.partner.view.setOriginScale(self.view.getOriginScale())
                start_x, start_y, end_x, end_y = self.view.GetExtend()
                self.partner.view.centerOn(start_x+(end_x-start_x)/2, start_y+(end_y-start_y)/2)

            def zoomEvent(scale, pos):
                changes2()
                self.signal_zoom.emit(scale, pos)
                self.partner.signal_zoom.emit(scale, pos)
            self.view.zoomEvent = zoomEvent
            self.view.panEvent = changes2
            changes2()

        else:
            #self.z_slider.setEnabled(True)
            def zoomEvent(scale, pos):
                self.partner.signal_zoom.emit(scale, pos)
            self.partner.view.zoomEvent = zoomEvent
            self.partner.view.panEvent = lambda *args: 0
            def zoomEvent(scale, pos):
                self.signal_zoom.emit(scale, pos)
            self.view.zoomEvent = zoomEvent
            self.view.panEvent = lambda *args: 0

    def setZCount(self, count):
        self.z_slider.setRange(0, count-1)
        self.z_count = count
        #self.z_slider.setMinimum(0)
        #self.z_slider.setMaximum(count-1)

    def getZ(self):
        return self.z_slider.value()

    def getZRange(self):
        minimum = int(max([self.getZ()-self.z_slider.active_range, 0]))
        maximum = int(min([self.getZ()+self.z_slider.active_range+1, self.z_count]))
        return slice(minimum, maximum)

    def zValueChanged(self, value):
        self.active.showImage()
        if self.partner is not None and self.input_linked.value() is True:
            if value != self.partner.z_slider.value():
                self.partner.z_slider.setValue(value)
                self.partner.z_slider.valueChanged.emit(value)
            if self.z_slider.active_range != self.partner.z_slider.active_range:
                self.partner.z_slider.active_range = self.z_slider.active_range
                self.partner.z_slider.activeRangeChanged.emit(value)

    def partner_zValueChanged(self, value):
        if self.input_linked.value() is True:
            if value != self.z_slider.value():
                self.z_slider.setValue(value)
                self.z_slider.valueChanged.emit(value)
            if self.partner.z_slider.active_range != self.z_slider.active_range:
                self.z_slider.active_range = self.partner.z_slider.active_range
                self.z_slider.activeRangeChanged.emit(value)
            #self.active.showImage()

    def getStack(self):
        return self.active.getStack()

    def validator(self):
        return self.active.validator()

    def validator_time(self):
        return self.active.validator_time()

    def getVoxelSize(self):
        if self.active is None:
            return None
        return self.active.getVoxelSize()

    def getTimeDelta(self):
        return self.active.getTimeDelta()

    def getStackParameters(self):
        return []
