import qtawesome as qta
from qtpy import QtCore, QtWidgets
from saenopy import Result
from typing import Tuple, List
import traceback


class TabModule(QtWidgets.QWidget):
    result: Result = None
    tab: QtWidgets.QTabWidget = None

    def __init__(self, parent: "BatchEvaluate"):
        super().__init__()
        self.parent = parent
        self.settings = self.parent.settings

        self.parent.result_changed.connect(self.resultChanged)
        self.parent.set_current_result.connect(self.setResult)
        self.parent.tab_changed.connect(self.tabChanged)

    current_result_plotted = False
    current_tab_selected = False
    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            self.current_tab_selected = True
            if self.current_result_plotted is False:
                self.update_display()
                self.current_result_plotted = True
        else:
            self.current_tab_selected = False

    def check_evaluated(self, result):
        return True

    def resultChanged(self, result: Result):
        """ called when the contents of result changed. Only update view if it is the currently displayed one. """
        if result is self.result:
            if self.tab is not None:
                for i in range(self.parent.tabs.count()):
                    if self.parent.tabs.widget(i) == self.tab.parent():
                       self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))
            if self.current_tab_selected is True:
                self.update_display()

    def setResult(self, result: Result):
        """ set a new active result object """
        self.current_result_plotted = False
        self.result = result

        if result is not None and getattr(self, "t_slider", None) is not None:
            data = result.get_data_structure()
            self.t_slider.setRange(0, data["time_point_count"] - 1)

        if self.tab is not None:
            for i in range(self.parent.tabs.count()):
                if self.parent.tabs.widget(i) == self.tab.parent():
                    self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))

        if self.current_tab_selected is True:
            self.update_display()

    def update_display(self):
        pass

