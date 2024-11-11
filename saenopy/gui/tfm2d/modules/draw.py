import numpy as np
import sys
import traceback
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
from qimage2ndarray import array2qimage

from saenopy.gui.common import QtShortCuts, QExtendedGraphicsView


class GraphicsItemEventFilter(QtWidgets.QGraphicsItem):
    def __init__(self, parent, command_object):
        super(GraphicsItemEventFilter, self).__init__(parent)
        self.commandObject = command_object
        self.active = True

    def paint(self, *args):
        pass

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 0, 0)

    def sceneEventFilter(self, scene_object, event):
        if not self.active:
            return False
        return self.commandObject.sceneEventFilter(event)


class DrawWindow(QtWidgets.QWidget):
    signal_mask_drawn = QtCore.Signal()
    alt_pressed = False

    def __init__(self, parent=None, layout=None):
        super().__init__(parent)
        if layout is not None:
            layout.addWidget(self)

        im = np.zeros((100, 100, 3), dtype=np.uint8)
        #im = plt.imread("/home/richard/PycharmProjects/pyTFM/example_data_for_pyTFM-master/python_tutorial/04before.tif")
        with QtShortCuts.QVBoxLayout(self) as main_layout:
            main_layout.setContentsMargins(0, 0, 0, 0)
            self.view1 = QExtendedGraphicsView.QExtendedGraphicsView().addToLayout()
            self.view1.setMinimumWidth(300)
            self.pixmap_image = QtWidgets.QGraphicsPixmapItem(self.view1.origin)
            self.pixmap_mask = QtWidgets.QGraphicsPixmapItem(self.view1.origin)

        self.pixmap_image.setPixmap(QtGui.QPixmap(array2qimage(im * 255)))
        self.view1.setExtend(im.shape[1], im.shape[0])

        self.scene_event_filter = GraphicsItemEventFilter(self.pixmap_image, self)
        self.scene_event_filter.commandObject = self
        self.pixmap_image.setAcceptHoverEvents(True)
        self.pixmap_image.installSceneEventFilter(self.scene_event_filter)

        self.DrawCursor = QtWidgets.QGraphicsPathItem(self.view1.origin)
        self.DrawCursor.setZValue(10)
        #self.DrawCursor.setVisible(False)

        self.full_image = Image.new("I", im.shape[:2][::-1])

        self.cursor_size = 10
        self.color = 1
        self.mask_opacity = 0.5

        self.palette = np.zeros((256, 4), dtype=np.uint8)
        self.palette[0, :] = [0, 0, 0, 0]
        self.palette[1, :] = [255, 0, 0, 255]
        self.palette[2, :] = [0, 255, 0, 255]
        self.palette[3, :] = [0, 0, 255, 255]
        self.palette[4, :] = [255, 255, 255, 255]

        self.UpdateDrawCursorDisplay()

    def setColor(self, color):
        self.color = color
        self.UpdateDrawCursorDisplay()

    def setMask(self, mask):
        self.full_image = Image.fromarray(mask.astype(np.uint8))
        im = np.asarray(self.full_image)
        im = self.palette[im]
        self.pixmap_mask.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view1.setExtend(mask.shape[1], mask.shape[0])
        self.changeOpacity(0)

    def setImage(self, im, shape=None):
        if shape is not None and (shape[0] != self.full_image.height or shape[1] != self.full_image.width):
            self.full_image = Image.new("I", shape[:2][::-1])
        self.pixmap_image.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.view1.setExtend(im.shape[1], im.shape[0])

    def setOpacity(self, value):
        self.mask_opacity = np.clip(value, 0, 1)
        self.pixmap_mask.setOpacity(self.mask_opacity)

    def changeOpacity(self, value: float) -> None:
        # alter the opacity by value
        self.mask_opacity += value
        # the opacity has to be maximally 1
        if self.mask_opacity >= 1:
            self.mask_opacity = 1
        # and minimally 0
        if self.mask_opacity < 0:
            self.mask_opacity = 0
        # set the opacity
        self.pixmap_mask.setOpacity(self.mask_opacity)

    def DrawLine(self, x1: float, x2: float, y1: float, y2: float, line_type: int = 1) -> None:
        size = self.cursor_size
        if line_type == 0:
            color = 0
        else:
            color = self.color#.index
        draw = ImageDraw.Draw(self.full_image)
        draw.line((x1, y1, x2, y2), fill=color, width=size + 1)
        draw.ellipse((x1 - size // 2, y1 - size // 2, x1 + size // 2, y1 + size // 2), fill=color)

        import numpy as np
        im = np.asarray(self.full_image)
        im = self.palette[im]
        self.pixmap_mask.setPixmap(QtGui.QPixmap(array2qimage(im)))
        self.changeOpacity(0)
        self.signal_mask_drawn.emit()

    def get_image(self):
        return np.asarray(self.full_image)

    def setCursorSize(self, size):
        self.cursor_size = size
        self.UpdateDrawCursorDisplay()

    def changeCursorSize(self, size):
        self.cursor_size += size
        self.UpdateDrawCursorDisplay()

    def UpdateDrawCursorDisplay(self) -> None:
        # update color and size of brush cursor
        draw_cursor_path = QtGui.QPainterPath()
        draw_cursor_path.addEllipse(-self.cursor_size * 0.5, -self.cursor_size * 0.5, self.cursor_size,
                                    self.cursor_size)
        pen = QtGui.QPen(QtGui.QColor(*self.palette[self.color if not self.alt_pressed else 4, 0:3]))
        pen.setCosmetic(True)
        self.DrawCursor.setPen(pen)
        self.DrawCursor.setPath(draw_cursor_path)

    def keyPressEvent(self, a0):
        if a0.key() == QtCore.Qt.Key_Alt:
            self.alt_pressed = True
            self.UpdateDrawCursorDisplay()

    def keyReleaseEvent(self, a0):
        if a0.key() == QtCore.Qt.Key_Alt:
            self.alt_pressed = False
            self.UpdateDrawCursorDisplay()

    def sceneEventFilter(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.GraphicsSceneMousePress and event.button() == QtCore.Qt.LeftButton:
            # store the coordinates
            self.last_pos = [event.pos().x(), event.pos().y()]
            paint = event.modifiers() != QtCore.Qt.AltModifier
            # add a first circle (so that even if the mouse isn't moved something is drawn)
            self.DrawLine(self.last_pos[0], self.last_pos[0] + 0.00001, self.last_pos[1], self.last_pos[1], paint)
            # accept the event
            return True
        if event.type() == QtCore.QEvent.GraphicsSceneMouseRelease and event.button() == QtCore.Qt.LeftButton:
            pass
        # Mouse move event to draw the stroke
        if event.type() == QtCore.QEvent.GraphicsSceneMouseMove:
            if (event.modifiers() == QtCore.Qt.AltModifier) != self.alt_pressed:
               self.alt_pressed = event.modifiers() == QtCore.Qt.AltModifier
               self.UpdateDrawCursorDisplay()
            pos = [event.pos().x(), event.pos().y()]
            # draw a line and store the position
            paint = event.modifiers() != QtCore.Qt.AltModifier
            self.DrawLine(pos[0], self.last_pos[0], pos[1],  self.last_pos[1], paint)
            self.last_pos = pos
            self.DrawCursor.setPos(event.pos())
            # accept the event
            return True
        # Mouse hover updates the color_under_cursor and displays the brush cursor
        if event.type() == QtCore.QEvent.GraphicsSceneHoverMove:
            if (event.modifiers() == QtCore.Qt.AltModifier) != self.alt_pressed:
                self.alt_pressed = event.modifiers() == QtCore.Qt.AltModifier
                self.UpdateDrawCursorDisplay()
            # move brush cursor
            self.DrawCursor.setPos(event.pos())
        if event.type() == QtCore.QEvent.GraphicsSceneWheel:
            try:  # PyQt 5
                angle = event.angleDelta().y()
            except AttributeError:  # PyQt 4
                angle = event.delta()
            # wheel with CTRL means changing the cursor size
            if event.modifiers() == QtCore.Qt.ControlModifier:
                if angle > 0:
                    self.changeCursorSize(+1)
                else:
                    self.changeCursorSize(-1)
                event.accept()
                return True
            # wheel with SHIFT means changing the opacity
            elif event.modifiers() == QtCore.Qt.ShiftModifier:
                if angle > 0:
                    self.changeOpacity(+0.1)
                else:
                    self.changeOpacity(-0.1)
                event.accept()
                return True
        # don't accept the event, so that others can accept it
        return False


class MinimalGui(QtWidgets.QWidget):
    def __init__(self, filename):
        super().__init__()
        with QtShortCuts.QVBoxLayout(self) as layout:
            self.draw = DrawWindow()
            im = plt.imread(filename)
            self.draw.setImage(im, shape=im.shape)
            layout.addWidget(self.draw)

            box = QtWidgets.QGroupBox("painting").addToLayout()
            with QtShortCuts.QVBoxLayout(box) as layout:
                self.slider_cursor_width = QtShortCuts.QInputNumber(None, "cursor width", 10, 1, 100, True, float=False)
                self.slider_cursor_width.valueChanged.connect(lambda x: self.draw.setCursorSize(x))
                self.slider_cursor_opacity = QtShortCuts.QInputNumber(None, "mask opacity", 0.5, 0, 1, True, float=True)
                self.slider_cursor_opacity.valueChanged.connect(lambda x: self.draw.setOpacity(x))
                with QtShortCuts.QHBoxLayout():
                    self.button_red = QtShortCuts.QPushButton(None, "tractions", lambda x: self.draw.setColor(1),
                                                              icon=qta.icon("fa5s.circle", color="red"))
                    self.button_green = QtShortCuts.QPushButton(None, "cell boundary", lambda x: self.draw.setColor(2),
                                                                icon=qta.icon("fa5s.circle", color="green"))
                QtWidgets.QLabel("hold 'alt' key for eraser").addToLayout()

app = None
def get_mask_using_gui(filename):
    global app
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        if sys.platform.startswith('win'):
            import ctypes
            myappid = 'fabrylab.saenopy.master'  # arbitrary string
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    window = MinimalGui(filename)
    window.show()
    res = app.exec_()
    return window.draw.get_image()

def main():
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform.startswith('win'):
        import ctypes
        myappid = 'fabrylab.saenopy.master'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    window = MinimalGui("/home/richard/PycharmProjects/pyTFM/example_data_for_pyTFM-master/python_tutorial/04before.tif")
    window.show()
    try:
        import pyi_splash

        # Update the text on the splash screen
        pyi_splash.update_text("PyInstaller is a great software!")
        pyi_splash.update_text("Second time's a charm!")

        # Close the splash screen. It does not matter when the call
        # to this function is made, the splash screen remains open until
        # this function is called or the Python program is terminated.
        pyi_splash.close()
    except (ImportError, RuntimeError):
        pass

    while True:
        try:
            res = app.exec_()
            break
        except Exception as err:
            traceback.print_traceback(err)
            QtWidgets.QMessageBox.critical(window, "Error", f"An Error occurred:\n{err}")
            continue
    sys.exit(res)


if __name__ == "__main__":
    print(get_mask_using_gui("/home/richard/PycharmProjects/pyTFM/example_data_for_pyTFM-master/python_tutorial/04before.tif"))
    print(get_mask_using_gui("/home/richard/PycharmProjects/pyTFM/example_data_for_pyTFM-master/clickpoints_tutorial/KO_analyzed/05bf_before.tif"))
    #main()
