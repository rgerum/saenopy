import sys
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui

import matplotlib.pyplot as plt

from saenopy.gui.common import QtShortCuts


def trace_function(func, data):
    import time
    last_time = 0
    def trace_lines(frame, event, arg):
        nonlocal t, last_time
        if event != 'line':
            return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        current_time = time.time()
        data.append([line_no, current_time-t])
        print(f'{func_name} line {line_no} {current_time-t:.2f} {current_time-last_time:.3f}')
        last_time = current_time

    t = 0
    def trace_calls(frame, event, arg):
        nonlocal t, last_time
        if event != 'call':
            return None
        co = frame.f_code
        func_name = co.co_name
        if func_name == 'write':
            # Ignore write() calls from print statements
            return None
        if func_name in TRACE_INTO:
            t = time.time()
            last_time = t
            # Trace into this function
            return trace_lines
        return None

    TRACE_INTO = [func]

    sys.settrace(trace_calls)


from matplotlib.figure import Figure

import ctypes

from qtpy import API_NAME as QT_API_NAME
if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar


class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        from matplotlib import _pylab_helpers
        plt.ioff()
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setStyleSheet("background-color:transparent;")
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.manager = FigureManager(self, 1)
        self.manager._cidgcf = self.figure

        """
        _pylab_helpers.Gcf.figs[num] = canvas.manager
        # get the canvas of the figure
        manager = _pylab_helpers.Gcf.figs[num]
        # set the size if it is defined
        if figsize is not None:
            _pylab_helpers.Gcf.figs[num].window.setGeometry(100, 100, figsize[0] * 80, figsize[1] * 80)
        # set the figure as the active figure
        _pylab_helpers.Gcf.set_active(manager)
        """
        _pylab_helpers.Gcf.set_active(self.manager)

    def setActive(self):
        from matplotlib import _pylab_helpers
        self.manager._cidgcf = self.figure
        _pylab_helpers.Gcf.set_active(self.manager)


def execute(func, *args, **kwargs):
    func(*args, **kwargs)
    import inspect
    code_lines = inspect.getsource(func).split("\n")[1:]
    indent = len(code_lines[0]) - len(code_lines[0].lstrip())
    code = "\n".join(line[indent:] for line in code_lines)
    for key, value in kwargs.items():
        if isinstance(value, str):
            code = code.replace(key, "'"+value+"'")
        else:
            code = code.replace(key, str(value))
    code = code.replace("self.canvas.draw()", "plt.show()")
    return code


def kill_thread(thread):
    """
    thread: a threading.Thread object
    """
    thread_id = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class Spoiler(QtWidgets.QWidget):
    def __init__(self, parent=None, title='', animationDuration=300):
        """
        References:
            # Adapted from c++ version
            http://stackoverflow.com/questions/32476006/how-to-make-an-expandable-collapsable-section-widget-in-qt
        """
        super(Spoiler, self).__init__(parent=parent)

        self.animationDuration = animationDuration
        self.toggleAnimation = QtCore.QParallelAnimationGroup()
        self.contentArea = QtWidgets.QScrollArea()
        self.headerLine = QtWidgets.QFrame()
        self.toggleButton = QtWidgets.QToolButton()
        self.mainLayout = QtWidgets.QGridLayout()

        toggleButton = self.toggleButton
        toggleButton.setStyleSheet("QToolButton { border: none; }")
        toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggleButton.setArrowType(QtCore.Qt.RightArrow)
        toggleButton.setText(str(title))
        toggleButton.setCheckable(True)
        toggleButton.setChecked(False)

        headerLine = self.headerLine
        headerLine.setFrameShape(QtWidgets.QFrame.HLine)
        headerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        headerLine.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

        self.contentArea.setStyleSheet("QScrollArea { background-color: white; border: none; }")
        self.contentArea.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        # start out collapsed
        self.contentArea.setMaximumHeight(0)
        self.contentArea.setMinimumHeight(0)
        # let the entire widget grow and shrink with its content
        toggleAnimation = self.toggleAnimation
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, b"minimumHeight"))
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self, b"maximumHeight"))
        toggleAnimation.addAnimation(QtCore.QPropertyAnimation(self.contentArea, b"maximumHeight"))
        # don't waste space
        mainLayout = self.mainLayout
        mainLayout.setVerticalSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        row = 0
        mainLayout.addWidget(self.toggleButton, row, 0, 1, 1, QtCore.Qt.AlignLeft)
        mainLayout.addWidget(self.headerLine, row, 2, 1, 1)
        row += 1
        mainLayout.addWidget(self.contentArea, row, 0, 1, 3)
        self.setLayout(self.mainLayout)

        def start_animation(checked):
            arrow_type = QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
            direction = QtCore.QAbstractAnimation.Forward if checked else QtCore.QAbstractAnimation.Backward
            toggleButton.setArrowType(arrow_type)
            self.toggleAnimation.setDirection(direction)
            self.toggleAnimation.start()

        self.toggleButton.clicked.connect(start_animation)

    def setContentLayout(self, contentLayout):
        # Not sure if this is equivalent to self.contentArea.destroy()
        self.contentArea.destroy()
        self.contentArea.setLayout(contentLayout)
        collapsedHeight = self.sizeHint().height() - self.contentArea.maximumHeight()
        contentHeight = contentLayout.sizeHint().height()
        for i in range(self.toggleAnimation.animationCount()-1):
            spoilerAnimation = self.toggleAnimation.animationAt(i)
            spoilerAnimation.setDuration(self.animationDuration)
            spoilerAnimation.setStartValue(collapsedHeight)
            spoilerAnimation.setEndValue(collapsedHeight + contentHeight)
        contentAnimation = self.toggleAnimation.animationAt(self.toggleAnimation.animationCount() - 1)
        contentAnimation.setDuration(self.animationDuration)
        contentAnimation.setStartValue(0)
        contentAnimation.setEndValue(contentHeight)


class CheckAbleGroup(QtWidgets.QWidget, QtShortCuts.EnterableLayout):
    value_changed = QtCore.Signal(bool)
    main_layout = None
    def __init__(self, parent=None, title='', animationDuration=300, url=None):
        super().__init__(parent=parent)

        self.headerLine = QtWidgets.QFrame()
        self.checkbox = QtWidgets.QCheckBox()
        self.toggleButton = QtWidgets.QToolButton()
        self.mainLayout = QtWidgets.QGridLayout()

        with QtShortCuts.QVBoxLayout(self) as self.main_layout:
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            with QtShortCuts.QHBoxLayout() as layout:
                layout.setContentsMargins(12, 0, 0, 0)
                self.toggleButton = QtShortCuts.QInputBool(None, "", True)
                self.label = QtWidgets.QPushButton(title).addToLayout()
                self.label.setStyleSheet("QPushButton { border: none; }")
                self.label.clicked.connect(self.toggle)

                headerLine = QtWidgets.QFrame().addToLayout()
                headerLine.setFrameShape(QtWidgets.QFrame.HLine)
                headerLine.setFrameShadow(QtWidgets.QFrame.Sunken)
                headerLine.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)

                #QtShortCuts.current_layout.addStretch()
                if url is not None:
                    self.label2 = QtWidgets.QPushButton(qta.icon("fa5s.question"), "").addToLayout()
                    self.label2.setToolTip("open the documentation in the browser")
                    #self.label2.setMaximumWidth(30)
                    self.label2.setStyleSheet("QPushButton { border: none; background: none; }")
                    self.label2.clicked.connect(lambda x: QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)))
            self.child_widget = QtWidgets.QWidget().addToLayout()
        self.layout = self
        self.value = self.toggleButton.value
        #self.setValue = self.toggleButton.setValue
        self.valueChanged = self.toggleButton.valueChanged
        self.toggleButton.valueChanged.connect(self.changedActive)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor("gray")))
        #p.setBrush(QtGui.QBrush(QtGui.QColor("gray")))
        top = 5
        p.drawRect(0, self.height()-1, self.width(), 0)
        p.drawRect(0, top, 0, self.height())
        p.drawRect(0, top, 7, 0)
        p.drawRect(self.width()-1, top, 0, self.height())
        super().paintEvent(ev)

    def toggle(self):
        self.setValue(not self.value())
        self.changedActive()

    def setValue(self, value):
        self.toggleButton.setValue(value)
        self.changedActive()

    def changedActive(self):
        self.value_changed.emit(self.value())
        self.child_widget.setEnabled(self.value())

    def addLayout(self, layout):
        if self.main_layout is None:
            self.setLayout(layout)
        else:
            self.child_widget.setLayout(layout)
        return layout


class ListWidget(QtWidgets.QListWidget):
    itemSelectionChanged2 = QtCore.Signal()
    itemChanged2 = QtCore.Signal()
    addItemClicked = QtCore.Signal()
    signal_act_copy_clicked = QtCore.Signal()
    signal_act_paste_clicked = QtCore.Signal()
    signal_act_paths_clicked = QtCore.Signal()

    data = []
    def __init__(self, layout, editable=False, add_item_button=False, color_picker=False, copy_params=False, allow_paste_callback=None):
        super().__init__()
        layout.addWidget(self)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.list2_context_menu)
        self.itemChanged.connect(self.list2_checked_changed)
        self.itemChanged = self.itemChanged2
        self.act_delete = QtWidgets.QAction(qta.icon("fa5.trash-alt"), "Remove", self)
        self.act_delete.triggered.connect(self.delete_item)

        self.act_color = None
        if color_picker is True:
            self.act_color = QtWidgets.QAction(qta.icon("fa5s.paint-brush"), "Change Color", self)
            self.act_color.triggered.connect(self.change_color)
        self.act_copy = None
        if copy_params is True:
            self.act_copy = QtWidgets.QAction(qta.icon("fa5.copy"), "Copy Parameters", self)
            self.act_copy.triggered.connect(self.signal_act_copy_clicked)
            self.act_paste = QtWidgets.QAction(qta.icon("fa5s.paste"), "Paste Parameters", self)
            self.act_paste.triggered.connect(self.signal_act_paste_clicked)
            self.allow_paste_callback = allow_paste_callback
            self.act_path = QtWidgets.QAction(qta.icon("mdi.folder-multiple-image"), "Adjust Paths", self)
            self.act_path.triggered.connect(self.signal_act_paths_clicked)

        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable
        if editable:
            self.flags |= QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsDragEnabled

        self.add_item_button = add_item_button
        self.addAddItem()
        self.itemSelectionChanged.connect(self.listSelected)
        self.itemSelectionChanged = self.itemSelectionChanged2

        self.itemClicked.connect(self.item_clicked)
        self.model().rowsMoved.connect(self.rowsMoved)

    def rowsMoved(self, parent, start, end, dest, row):
        if row == self.count():
            if self.add_item is not None:
                self.takeItem(self.count() - 2)
            self.addAddItem()
            return
        if row > start:
            row -= 1
        self.data.insert(row, self.data.pop(start))

    def item_clicked(self, item):
        if item == self.add_item:
            self.addItemClicked.emit()

    def listSelected(self):
        if self.no_list_change is True:
            return
        self.no_list_change = True
        if self.currentItem() == self.add_item:
            self.no_list_change = False
            return
        self.itemSelectionChanged.emit()
        self.no_list_change = False

    add_item = None
    def addAddItem(self):
        if self.add_item_button is False:
            return
        if self.add_item is not None:
            del self.add_item
        self.add_item = QtWidgets.QListWidgetItem(qta.icon("fa5s.plus"), self.add_item_button, self)
        self.add_item.setFlags(QtCore.Qt.ItemIsEnabled)

    def list2_context_menu(self, position):
        if self.currentItem() and self.currentItem() != self.add_item:
            # context menu
            menu = QtWidgets.QMenu()

            if self.act_color is not None:
                menu.addAction(self.act_color)
            if self.act_copy is not None:
                menu.addAction(self.act_copy)
                self.act_paste.setDisabled(not self.allow_paste_callback())
                menu.addAction(self.act_paste)
                menu.addAction(self.act_path)

            menu.addAction(self.act_delete)

            # open menu at mouse click position
            if menu:
                menu.exec_(self.viewport().mapToGlobal(position))

    def change_color(self):
        import matplotlib as mpl
        index = self.currentRow()

        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*[int(i) for i in mpl.colors.to_rgb(self.data[index][3])]))\
        # if a color is set, apply it
        if color.isValid():
            self.data[index][3] = "#%02x%02x%02x" % color.getRgb()[:3]
            self.item(index).setIcon(qta.icon("fa5s.circle", options=[dict(color=color)]))

    def delete_item(self):
        index = self.currentRow()
        self.data.pop(index)
        self.takeItem(index)
        self.setCurrentRow(index)

    def setData(self, data):
        self.no_list_change = True
        self.data = data
        self.clear()
        for d, checked, _, color in data:
            self.customAddItem(d, checked, color)

        self.addAddItem()
        self.no_list_change = False

    no_list_change = False
    def list2_checked_changed(self, item):
        if self.no_list_change is True:
            return
        data = self.data
        for i in range(len(data)):
            item = self.item(i)
            data[i][0] = item.text()
            data[i][1] = item.checkState()
        self.itemChanged.emit()

    def addData(self, d, checked, extra=None, color=None):
        # check if the element is already in the list
        for element in self.data:
            if element[0] == d:
                return None
        self.no_list_change = True
        if self.add_item is not None:
            self.takeItem(self.count()-1)
        self.data.append([d, checked, extra, color])
        item = self.customAddItem(d, checked, color)
        self.addAddItem()
        self.no_list_change = False
        return item

    def customAddItem(self, d, checked, color):
        item = QtWidgets.QListWidgetItem(d, self)
        if color is not None:
            item.setIcon(qta.icon("fa5s.circle", options=[dict(color=color)]))
        item.setFlags(self.flags)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item


from multiprocessing import Process, Queue


class SignalReturn:
    pass

def call_func(func: callable, queue_in: Queue, queue_out: Queue):
    args = queue_in.get()
    kwargs = queue_in.get()
    returns = func(queue_out, *args, **kwargs)
    queue_out.put(SignalReturn())
    queue_out.put(returns)


class PseudoPipe:
    def __init__(self, progress_signal):
        self.progress_signal = progress_signal
    def put(self, data):
        self.progress_signal.emit(data)

import threading

class ConciseRobustResult(threading.Thread):
    def run(self):
        try:
            if self._target is not None:
                self.result = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


class ProcessSimple:
    thread = None
    process = None

    def __init__(self, target, args=[], kwargs={}, progress_signal=None, use_thread=False):
        self.args = args
        self.kwargs = kwargs
        self.progress_signal = progress_signal
        if use_thread:
            self.thread = ConciseRobustResult(target=target, args=tuple([PseudoPipe(progress_signal)]+list(args)), kwargs=kwargs)
        else:
            self.queue_in = Queue()
            self.queue_out = Queue()
            self.process = Process(target=call_func, args=(target, self.queue_in, self.queue_out))

    def start(self):
        if self.thread is not None:
            self.thread.start()
        else:
            self.process.start()
            self.queue_in.put(self.args)
            self.queue_in.put(self.kwargs)

    def join(self):
        if self.thread:
            self.thread.join(timeout=10)
            return self.thread.result
        while True:
            result = self.queue_out.get()
            if isinstance(result, SignalReturn):
                result = self.queue_out.get()
                break
            elif self.progress_signal is not None:
                self.progress_signal.emit(result)
        self.process.join()
        return result


# Step 1: Create a worker class
class Worker(QtCore.QObject):
    finished = QtCore.Signal(object)
    progress = QtCore.Signal(object)

    def __init__(self, func, args, kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Long-running task."""
        p = ProcessSimple(self.func, self.args, self.kwargs, progress_signal=self.progress)
        p.start()
        result = p.join()
        #for i in range(5):
        #    time.sleep(1)
        #    self.progress.emit(i + 1)
        self.finished.emit(result)

class QProcess(QtCore.QObject):
    result = None
    finished = QtCore.Signal()
    progress = QtCore.Signal(object)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        # Step 2: Create a QThread object
        self.thread = QtCore.QThread()
        # Step 3: Create a worker object
        self.worker = Worker(func, args, kwargs)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        #self.worker.finished.connect(self.thread.quit)
        #self.worker.finished.connect(self.worker.deleteLater)
        #self.worker.finished.connect(self.finished)
        self.worker.finished.connect(self.set_result)
        self.worker.progress.connect(self.progress)
        self.thread.finished.connect(self.thread.deleteLater)
        #self.worker.progress.connect(self.reportProgress)

        # Step 6: Start the thread
        self.thread.start()

    def set_result(self, result):
        self.result = result
        self.thread.quit()
        self.worker.deleteLater()
        self.finished.emit()


def wrap_string(func):
    def call(*args, **kwargs):
        new_path = func(*args, **kwargs)
        if new_path:
            if isinstance(new_path, tuple):
                new_path = new_path[0]
            else:
                new_path = str(new_path)
        return new_path
    return call

QtWidgets.QFileDialog.getSaveFileName = wrap_string(QtWidgets.QFileDialog.getSaveFileName)
QtWidgets.QFileDialog.getOpenFileName = wrap_string(QtWidgets.QFileDialog.getOpenFileName)