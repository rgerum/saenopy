from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from qtpy import QtCore, QtGui, QtWidgets

def create_widget(element):
    # Create a widget based on the element type
    if element['type'] == 'QVBoxLayout':
        layout = QVBoxLayout()
    elif element['type'] == 'QHBoxLayout':
        layout = QHBoxLayout()
    elif element['type'] == 'QLabel':
        widget = QLabel()
        if element['props'] and 'text' in element['props']:
            widget.setText(element['props']['text'])
        return widget
    elif element['type'] == 'QTextEdit':
        widget = QtWidgets.QTextEdit()
        if element['props'] and 'text' in element['props']:
            widget.setText(element['props']['text'])
        if element['props'] and 'onTextChanged' in element["props"]:
            widget.textChanged.connect(lambda : element['props']['onTextChanged'](widget.toPlainText()))
        return widget
    elif element['type'] == 'QPushButton':
        widget = QPushButton()
        if element['props'] and 'text' in element['props']:
            widget.setText(element['props']['text'])
        # Connect the button's click event to the provided onClick function
        if element['props'] and 'onClick' in element['props']:
            widget.clicked.connect(element['props']['onClick'])
        return widget
    else:
        return None  # Unknown type

    # Add children to the layout
    if 'children' in element:
        for child in element['children']:
            child_widget = create_widget(child)
            if isinstance(child_widget, QWidget):
                layout.addWidget(child_widget)
            else:
                layout.addLayout(child_widget)

    return layout

def update_widget(widget, new_element):
    if isinstance(widget, QLabel):
        if new_element['props'] and 'text' in new_element['props']:
            widget.setText(new_element['props']['text'])
    elif isinstance(widget, QtWidgets.QTextEdit):
        widget.textChanged.disconnect()
        if new_element['props'] and 'text' in new_element['props']:
            if widget.toPlainText() != new_element['props']['text']:
                widget.setText(new_element['props']['text'])
        if new_element['props'] and 'onTextChanged' in new_element["props"]:
            widget.textChanged.connect(lambda : new_element['props']['onTextChanged'](widget.toPlainText()))
    elif isinstance(widget, QPushButton):
        if new_element['props'] and 'text' in new_element['props']:
            widget.setText(new_element['props']['text'])
        if new_element['props'] and 'onClick' in new_element['props']:
            widget.clicked.disconnect()  # Disconnect previous handler
            widget.clicked.connect(new_element['props']['onClick'])
    elif isinstance(widget, QVBoxLayout) or isinstance(widget, QHBoxLayout):
        # Update children
        i = 0
        for i, child in enumerate(new_element['children']):
            if i < widget.count():
                if widget.itemAt(i).widget() is not None:
                    update_widget(widget.itemAt(i).widget(), child)
                else:
                    update_widget(widget.itemAt(i), child)
            else:
                # Add new child
                child_widget = create_widget(child)
                if isinstance(child_widget, QWidget):
                    widget.addWidget(child_widget)
                else:
                    widget.addLayout(child_widget)
        for j in range(i+1, widget.count()):
            if widget.itemAt(j).widget() is not None:
                widget.itemAt(j).widget().deleteLater()
            else:
                delete_layout(widget.itemAt(j))

def delete_layout(layout):
    for i in reversed(range(layout.count())):
        widget_item = layout.itemAt(i)
        if widget_item is not None and widget_item.widget() is not None:
            widget_item.widget().deleteLater()
        else:
            delete_layout(widget_item)

def update_ui(old_structure, new_structure, main_layout):
    # Clear the old layout
    for i in range(main_layout.count()):
        widget_item = main_layout.itemAt(i)
        update_widget(widget_item, new_structure)
        break

# Example structure
structure = {'children': [{'children': [],
               'props': {'text': 'Hello, World!'},
               'type': 'QLabel'},
              {'children': [],
               'props': {'onClick': lambda x: print('Button clicked'),
                         'text': 'Click Me'},
               'type': 'QPushButton'},
              {'children': [{'children': [],
                             'props': {'className': 'my-class', 'text': 'foo'},
                             'type': 'QLabel'},
                            {'children': [],
                             'props': {'className': 'my-class', 'text': 'bar'},
                             'type': 'QLabel'},
                            {'children': [],
                             'props': {'className': 'my-class', 'text': 'baz'},
                             'type': 'QLabel'}],
               'props': {'className': 'my-class'},
               'type': 'QHBoxLayout'}],
 'props': {'className': 'my-class'},
 'type': 'QVBoxLayout'}


def createQtApp(structure):
    # Create the application and main window
    app = QApplication([])
    main_window = QWidget()
    main_layout = QVBoxLayout()
    main_window.setLayout(main_layout)

    # Create the main widget from the structure
    main_widget = create_widget(structure)
    main_layout.addLayout(main_widget)

    return app, main_window, main_layout

