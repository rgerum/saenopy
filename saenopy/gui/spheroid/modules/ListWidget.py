import qtawesome as qta

from qtpy import QtCore, QtWidgets, QtGui

settings = QtCore.QSettings("Saenopy", "Saenopy")


class ListWidget(QtWidgets.QListWidget):
    itemSelectionChanged2 = QtCore.Signal()
    itemChanged2 = QtCore.Signal()
    addItemClicked = QtCore.Signal()

    data = []
    def __init__(self, layout, editable=False, add_item_button=False, color_picker=False):
        super().__init__()
        layout.addWidget(self)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.list2_context_menu)
        self.itemChanged.connect(self.list2_checked_changed)
        self.itemChanged = self.itemChanged2
        self.act_delete = QtWidgets.QAction(qta.icon("fa5s.trash-alt"), "Delete", self)
        self.act_delete.triggered.connect(self.delete_item)

        self.act_color = None
        if color_picker is True:
            self.act_color = QtWidgets.QAction(qta.icon("fa5s.paint-brush"), "Change Color", self)
            self.act_color.triggered.connect(self.change_color)

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

            menu.addAction(self.act_delete)

            # open menu at mouse click position
            if menu:
                menu.exec_(self.viewport().mapToGlobal(position))

    def change_color(self):
        import matplotlib as mpl
        index = self.currentRow()

        # get new color from color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*mpl.colors.to_rgb(self.data[index][3])))
        # if a color is set, apply it
        if color.isValid():
            self.data[index][3] = "#%02x%02x%02x" % color.getRgb()[:3]
            self.item(index).setIcon(qta.icon("fa5.circle", options=[dict(color=color)]))

    def delete_item(self):
        index = self.currentRow()
        self.data.pop(index)
        self.takeItem(index)

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
            item.setIcon(qta.icon("fa5.circle", options=[dict(color=color)]))
        item.setFlags(self.flags)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        return item
