import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np


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


class SpeedSlider():
    def __init__(self, parent, ax, extent, end):
        self.parent = parent
        self.extent = extent
        self.end = end
        self.centerx = (extent[1]-extent[0])/2+extent[0]
        self.centery = (extent[3]-extent[2])/2+extent[2]
        self.path3 = matplotlib.patches.Rectangle((extent[0], extent[2]), extent[1]-extent[0], extent[3]-extent[2], facecolor="gray", edgecolor="black")
        ax.add_patch(self.path3)
        self.im3 = ax.imshow(np.arange(0, 255)[None, :], cmap="gray", extent=extent, zorder=1)
        self.im3.set_clip_path(self.path3)

        self.line_hist_grabber1b, = ax.step([self.centerx], [self.centery], "C3|", lw=2, ms=5)

    def get_speed(self, event):
        return (event.xdata - self.centerx) / 10

    def get_clicked(self, event):
        if event.xdata is None:
            return 0
        if event.ydata < 0:
            if -0.2 < event.ydata <= -0.1 and self.extent[0] <= event.xdata < self.extent[1]:
                return 12

    def update_line(self, hovered, value):
        self.line_hist_grabber1b.set_markersize(8 if hovered else 5)
        if self.end is False:
            self.im3.set_extent([self.centerx - value * 10, self.centerx + 10 - value * 10, self.extent[2], self.extent[3]])
        else:
            self.im3.set_extent(
                [self.centerx - 10 - (value - 1) * 10, self.centerx - (value - 1) * 10, self.extent[2], self.extent[3]])


class HistSlider:
    def __init__(self, parent, ax, extent):
        self.parent = parent
        self.extent = extent
        self.centery = (extent[3] - extent[2]) / 2 + extent[2]

        self.path3 = matplotlib.patches.Rectangle((0, 0), 1, 1, facecolor="none")
        ax.add_patch(self.path3)

        self.line_hist, = ax.step([0, 1], [1, 1], '-', color="gray", lw=1, alpha=0.5)
        self.line_hist.set_clip_path(self.path3)
        self.line_hist_grabber1, = ax.step([extent[0]], [self.centery], "C3", marker=9, lw=2, ms=5)
        #self.line_hist_grabber1b, = ax.step([0.25], [-0.05-0.1], "C3|", lw=2, ms=5)
        self.line_hist_grabber2, = ax.step([extent[1]], [self.centery], "C3", marker=8, lw=2, ms=5)
        #self.line_hist_grabber2b, = ax.step([0.75], [-0.05 - 0.1], "C3|", lw=2, ms=5)
        self.im2 = ax.imshow(np.arange(0, 255)[None, :], cmap="gray", extent=extent, zorder=1)

    def get_clicked(self, event, minx, maxx):
        if self.extent[2] < event.ydata < self.extent[3]:
            if event.xdata < 0.5:
                if abs(maxx - event.xdata) < 0.04:
                    return 2
                if abs(minx - event.xdata) < 0.04:
                    return 1
            else:
                if abs(minx - event.xdata) < 0.04:
                    return 1
                if abs(maxx - event.xdata) < 0.04:
                    return 2

    def update_line(self, hovered1, hovered2, minx, maxx, hist_bins):
        self.line_hist_grabber1.set_xdata([minx])
        self.line_hist_grabber1.set_markersize(8 if hovered1 else 5)
        self.line_hist_grabber2.set_xdata([maxx])
        self.line_hist_grabber2.set_markersize(8 if hovered2 else 5)

        self.line_hist.set_xdata((hist_bins-minx)/(maxx-minx))

class PolygonInteractor:
    grabbed = 0
    hovered = 0

    def __init__(self, parent, ax):
        self.parent = parent
        self.ax = ax
        self.a1 = 0.1
        self.a2 = 0.5
        self.a3 = 1
        self.minx = 0
        self.maxx = 1
        canvas = ax.figure.canvas
        self.cmap = "Reds"
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        dx = 0.1
        ax.set_xlim(-dx, 1 + dx)
        ax.set_ylim(-dx-0.12, 1 + dx)
        self.path = matplotlib.patches.Polygon([[0, 0], [1, 0], [1, 1]], alpha=0)
        ax.add_patch(self.path)
        self.im = ax.imshow(np.arange(0, 255)[None, :], cmap="Greens", extent=[0, 1, 0, 1], zorder=1)
        self.im.set_clip_path(self.path)
        self.line2a, = ax.plot([0, 1], [1, 1], 'o-k', lw=0.8, ms=3)
        self.line2b, = ax.plot([0, 1], [1, 1], 'o-k', lw=.8, ms=3)
        self.line, = ax.plot([0, 1], [0, 1], '-', lw=2, color=plt.get_cmap("Greens")(0.5))
        self.line2c, = ax.plot([0, 1], [1, 1], 'o--k', lw=0.8, ms=3)
        self.line2_hover, = ax.plot([0, 1], [1, 1], 'o-k', lw=2, ms=5)

        self.hist_slider = HistSlider(self, ax, [0, 1, -0.08-0.02, -0.02-0.02])

        self.hist_bins = np.arange(0, 1, 10)

        self.speed_slider1 = SpeedSlider(self, ax, [0, 0.4, -0.08-0.12, -0.02-0.12], False)
        self.speed_slider2 = SpeedSlider(self, ax, [0.6, 1, -0.08-0.12, -0.02-0.12], True)
        if 0:
            self.path3 = matplotlib.patches.Rectangle((0, -0.08-0.1), 0.5, 0.06, facecolor="gray", edgecolor="black")
            ax.add_patch(self.path3)
            self.im3 = ax.imshow(np.arange(0, 255)[None, :], cmap="gray", extent=[0, 1, -0.08-0.1, -0.02-0.1], zorder=1)
            self.im3.set_clip_path(self.path3)

            self.path4 = matplotlib.patches.Rectangle((0.5, -0.08 - 0.1), 0.5, 0.06, facecolor="gray", edgecolor="black")
            ax.add_patch(self.path4)
            self.im4 = ax.imshow(np.arange(0, 255)[None, :], cmap="gray", extent=[0, 1, -0.08 - 0.1, -0.02 - 0.1], zorder=1)
            self.im4.set_clip_path(self.path4)


        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

        self.timer = QtCore.QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.timercall)

        self.update_line()
        self.set_cmap("pink")

    original_min = 0
    original_max = 1
    def set_im(self, im):
        self.original_min = im.min()
        self.original_max = im.max()
        bins = np.linspace(self.original_min, self.original_max, 128)
        hist, bins = np.histogram(im, bins)
        hist = np.arcsinh(hist)
        self.hist = hist
        self.hist_bins = np.linspace(0, 1, 128-1)
        self.hist_slider.im2.set_data(hist[None]*255/np.max(hist))
        self.speed_slider1.im3.set_data(hist[None]*255/np.max(hist))
        self.speed_slider2.im3.set_data(hist[None]*255/np.max(hist))
        #self.im3.set_data(hist[None]*255/np.max(hist))
        #self.im4.set_data(hist[None]*255/np.max(hist))

        self.hist_slider.line_hist.set_data([
            self.hist_bins, hist/np.max(hist),
        ])

        self.update_line()

    def get_range(self):
        return self.minx, self.maxx
        return self.original_min + (self.original_max - self.original_min) * self.minx, \
               self.original_min + (self.original_max - self.original_min) * self.maxx


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

        self.hist_slider.update_line(self.hovered == 10, self.hovered == 11, self.minx, self.maxx, self.hist_bins)

        #if self.grabbed == 10:
        self.speed_slider1.update_line(self.hovered == 12, self.minx)
        self.speed_slider2.update_line(self.hovered == 13, self.maxx)
        if 0:
            self.line_hist_grabber1b.set_markersize(8 if self.hovered == 12 else 5)
            self.im3.set_extent([0.25 - self.minx * 10, 0.25 + 10 - self.minx*10, -0.08-0.1, -0.02-0.1])

            self.line_hist_grabber2b.set_markersize(8 if self.hovered == 13 else 5)
            self.im4.set_extent([0.75 - 10 - (self.maxx-1) * 10, 0.75 - (self.maxx-1) * 10, -0.08 - 0.1, -0.02 - 0.1])
        #self.line_hist_grabber1b.set_xdata([self.start_min_x + (self.minx - self.start_min_x)*10])

        self.canvas.draw()

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.grabbed = self.get_clicked(event)
        self.start_eventx = event.xdata
        self.start_min_x = self.minx
        self.start_max_x = self.maxx
        if self.grabbed == 12 or self.grabbed == 13:
            self.speed = 0
            self.timer.start()

    def timercall(self):
        if self.grabbed == 12:
            self.minx = np.clip(self.minx + self.speed, 0, self.maxx - 0.01)
            self.update_line()
        if self.grabbed == 13:
            self.maxx = np.clip(self.maxx+self.speed, self.minx+0.01, 1)
            self.update_line()

    def get_clicked(self, event):
        if event.xdata is None:
            return 0

        if self.speed_slider1.get_clicked(event):
            return 12
        if self.speed_slider2.get_clicked(event):
            return 13

        if self.hist_slider.get_clicked(event, self.minx, self.maxx) == 1:
            return 10
        if self.hist_slider.get_clicked(event, self.minx, self.maxx) == 2:
            return 11

        m = 1 / (4 * self.a1 / self.a3)
        t = 0.5 * self.a3 - m * self.a2
        x1 = self.a2 - 0.3 * np.sqrt(1 / (1 + m ** 2))
        x2 = self.a2 + 0.3 * np.sqrt(1 / (1 + m ** 2))
        if dist_line_to_point([0, t], [1, m + t], [event.xdata, event.ydata]) < 0.04 and \
            x1 - 0.04 <= event.xdata < x2 + 0.04:
            return 1
        elif abs(event.xdata - self.a2) < 0.04 and -0.04 <= event.ydata < self.a3/2 + 0.04:
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
        self.timer.stop()

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
            self.parent.new_values(self.a1, self.a2, self.a3, self.minx, self.maxx)
            self.update_line()
        if self.grabbed == 3:
            if event.ydata:
                self.a3 = np.clip(event.ydata, 0, 1)
                self.parent.new_values(self.a1, self.a2, self.a3, self.minx, self.maxx)
                self.update_line()
        if self.grabbed == 2:
            if event.xdata:
                self.a2 = np.clip(event.xdata, 0, 1)
                self.parent.new_values(self.a1, self.a2, self.a3, self.minx, self.maxx)
                self.update_line()
        if self.grabbed == 10:
            if event.xdata:
                self.minx = np.clip(event.xdata, 0, self.maxx-0.05)
                self.parent.new_values(self.a1, self.a2, self.a3, self.minx, self.maxx)
                self.update_line()
        if self.grabbed == 11:
            if event.xdata:
                self.maxx = np.clip(event.xdata, self.minx+0.05, 1)
                self.parent.new_values(self.a1, self.a2, self.a3, self.minx, self.maxx)
                self.update_line()

        if self.grabbed == 12:
            self.speed = self.speed_slider1.get_speed(event)
        if self.grabbed == 13:
            self.speed = self.speed_slider2.get_speed(event)

        if self.grabbed == 0 and self.get_clicked(event) != self.hovered:
            self.hovered = self.get_clicked(event)
            self.update_line()

from qtpy import QtWidgets, QtCore
from saenopy.gui.common.gui_classes import MatplotlibWidget
class SigmoidWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float, float, float, float, float)
    editFinished = QtCore.Signal()
    minx = 0
    maxx = 1
    a1 = 0.1
    a2 = 0.5
    a3 = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MatplotlibWidget(self)
        QtWidgets.QHBoxLayout(self)
        size = 300
        #self.setMinimumWidth(size)
        #self.setMaximumWidth(size)
        #self.setMinimumHeight(size)
        #self.setMaximumHeight(size)
        self.layout().addWidget(self.canvas)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.canvas.figure.axes[0].set_position([0, 0, 1, 1])
        self.canvas.figure.axes[0].set_facecolor("none")
        self.p = PolygonInteractor(self, self.canvas.figure.axes[0])
        #self.canvas.

    def new_values(self, a1, a2, a3, minx, maxx):
        self.a1, self.a2, self.a3, self.minx, self.maxx = a1, a2, a3, minx, maxx
        self.valueChanged.emit(a1, a2, a3, self.p.get_range()[0], self.p.get_range()[1])


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SigmoidWidget()
    im = plt.imread(
        "/home/richard/.local/share/saenopy/1_ClassicSingleCellTFM/Deformed/Mark_and_Find_001/Pos004_S001_z000_ch00.tif")
    window.p.set_im(im)
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
