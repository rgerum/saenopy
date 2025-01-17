from saenopy.gui.common.TabModule import TabModule
import os

from qtpy import QtCore, QtWidgets, QtGui

import traceback

from saenopy import get_stacks
from saenopy.gui.common import QtShortCuts

class TabCellImage(TabModule):

    def __init__(self, parent=None):
        super().__init__(parent)
        with self.parent.tabs.createTab("Cell Image") as self.tab:
            pass

    def tabChanged(self, tab):
        if self.tab is not None and self.tab.parent() == tab:
            im = self.result.get_image(-1)
            self.parent.draw.setImage(im, self.result.shape)
