import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist


def get_transparency(a, b, alpha):
    x1 = np.linspace(0, 1, 21)

    def sigmoid(x):
        return 1 / (1 + np.exp(-((x - b) / a)))
        # return 1 / (1 + np.exp(-(x / a) - b))

    x = sigmoid(x1)
    opacity = 1.0 * alpha * x
    return opacity


def dist_line_to_point(p1, p2, p0):
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p0
    return np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class PolygonInteractor:
    grabbed = 0
    hovered = 0

    def __init__(self, parent, ax):
        self.parent = parent
        self.ax = ax
        self.a1 = 0.1
        self.a2 = 0.5
        self.a3 = 1
        canvas = ax.figure.canvas
        self.cmap = "Reds"
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        dx = 0.1
        ax.set_xlim(-dx, 1 + dx)
        ax.set_ylim(-dx, 1 + dx)
        self.path = matplotlib.patches.Polygon([[0, 0], [1, 0], [1, 1]], alpha=0)
        ax.add_patch(self.path)
        self.im = ax.imshow(np.arange(0, 255)[None, :], cmap="Greens", extent=[0, 1, 0, 1], zorder=1)
        self.im.set_clip_path(self.path)
        self.line2a, = ax.plot([0, 1], [1, 1], 'o-k', lw=0.8, ms=3)
        self.line2b, = ax.plot([0, 1], [1, 1], 'o-k', lw=.8, ms=3)
        self.line, = ax.plot([0, 1], [0, 1], '-', lw=2, color=plt.get_cmap("Greens")(0.5))
        self.line2c, = ax.plot([0, 1], [1, 1], 'o--k', lw=0.8, ms=3)
        self.line2_hover, = ax.plot([0, 1], [1, 1], 'o-k', lw=2, ms=5)

        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas
        self.update_line()
        self.set_cmap("pink")

    def set_cmap(self, color):
        self.im.set_cmap(color)
        self.line.set_color(plt.get_cmap(color)(0.5))
        self.canvas.draw()

    def update_line(self):
        opacity = get_transparency(self.a1, self.a2, self.a3)
        self.line.set_data([np.linspace(0, 1, len(opacity)), opacity])
        self.path.set_xy(list(np.array(self.line.get_data()).T) + [[1, 0], [0, 0]])
        m = 1 / (4 * self.a1 / self.a3)
        t = 0.5 * self.a3 - m * self.a2
        x1 = np.clip((-t) / m, 0, 1)
        x2 = np.clip((self.a3 - t) / m, 0, 1)

        x1 = self.a2 - 0.3 * np.sqrt(1/(1+m**2))
        x2 = self.a2 + 0.3 * np.sqrt(1/(1+m**2))

        self.line2a.set_data([
            [0, 1],
            [self.a3, self.a3]])
        self.line2b.set_data([
            [self.a2, self.a2],
            [0, self.a3/2]])
        self.line2c.set_data([
            [x1, x2],
            [x1 * m + t, x2 * m + t]])
        if self.hovered == 1:
            self.line2_hover.set_data([
                [x1, x2],
                [x1 * m + t, m * x2 + t]
            ])
        if self.hovered == 2:
            self.line2_hover.set_data([
                [self.a2, self.a2],
                [0, self.a3/2]
            ])
        if self.hovered == 3:
            self.line2_hover.set_data([
                [0, 1],
                [self.a3, self.a3]
            ])
        if self.hovered == 0:
            self.line2_hover.set_data([[], []])
        self.canvas.draw()

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.grabbed = self.get_clicked(event)

    def get_clicked(self, event):
        if event.xdata is None:
            return 0
        m = 1 / (4 * self.a1 / self.a3)
        t = 0.5 * self.a3 - m * self.a2
        if dist_line_to_point([0, t], [1, m + t], [event.xdata, event.ydata]) < 0.04:
            return 1
        elif abs(event.xdata - self.a2) < 0.04:
            return 2
        elif abs(event.ydata - self.a3) < 0.04:
            return 3
        return 0

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if event.button != 1:
            return
        if self.grabbed:
            self.parent.editFinished.emit()
        self.grabbed = 0

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.xdata is None:
            return
        if self.grabbed == 1:
            dx = (event.xdata - self.a2)
            dy = (event.ydata - self.a3 / 2)
            if (dy < 0 and dx > 0) or (dy > 0 and dx < 0):
                self.a1 = 1e6
            else:
                m = dy / dx
                self.a1 = 1 / (4 * m / self.a3)
                self.a1 = np.clip(self.a1, 1e-3, 1e6)
            self.parent.new_values(self.a1, self.a2, self.a3)
            self.update_line()
        if self.grabbed == 3:
            if event.ydata:
                self.a3 = np.clip(event.ydata, 0, 1)
                self.parent.new_values(self.a1, self.a2, self.a3)
                self.update_line()
        if self.grabbed == 2:
            if event.xdata:
                self.a2 = np.clip(event.xdata, 0, 1)
                self.parent.new_values(self.a1, self.a2, self.a3)
                self.update_line()
        if self.grabbed == 0 and self.get_clicked(event) != self.hovered:
            self.hovered = self.get_clicked(event)
            self.update_line()

from qtpy import QtWidgets, QtCore, QtGui
from saenopy.gui.gui_classes import MatplotlibWidget
class SigmoidWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float, float, float)
    editFinished = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MatplotlibWidget(self)
        QtWidgets.QHBoxLayout(self)
        self.setMinimumWidth(80)
        self.setMaximumWidth(80)
        self.setMinimumHeight(80)
        self.setMaximumHeight(80)
        self.layout().addWidget(self.canvas)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.canvas.figure.axes[0].set_position([0, 0, 1, 1])
        self.canvas.figure.axes[0].set_facecolor("none")
        self.p = PolygonInteractor(self, self.canvas.figure.axes[0])

    def new_values(self, a1, a2, a3):
        self.a1, self.a2, self.a3 = a1, a2, a3
        self.valueChanged.emit(a1, a2, a3)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SigmoidWidget()
    window.show()
    app.exec_()

if 0:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    theta = np.arange(0, 2 * np.pi, 0.1)
    r = 1.5

    xs = r * np.cos(theta)
    ys = r * np.sin(theta)

    poly = Polygon(np.column_stack([xs, ys]), animated=True)

    fig, ax = plt.subplots()
    ax.add_patch(poly)
    p = PolygonInteractor(ax)

    # ax.set_title('Click and drag a point to move it')
    # ax.set_xlim((-2, 2))
    # ax.set_ylim((-2, 2))
    plt.show()
