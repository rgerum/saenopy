from qtpy import QtWidgets
from pathlib import Path
import re

from saenopy.gui.common import QtShortCuts


def start_path_change(parent, result):
    path_editor = PathEditor(parent, result)
    if not path_editor.exec():
        return

    path_changer = PathChanger(result.template, path_editor.input_folder.value())
    for stack in result.stacks:
        path_changer.stack_change(stack)
    result.template = path_changer.change_path(result.template)

    if path_editor.input_folder2 is not None:
        path_changer = PathChanger(result.stack_reference.template, path_editor.input_folder2.value())
        path_changer.stack_change(result.stack_reference)

    if path_editor.input_pack.value():
        for stack in result.stacks:
            stack.pack_files()
        if result.stack_reference:
            result.stack_reference.pack_files()

    if path_editor.input_save.value():
        result.save()
        #print("saved")
        return


class PathEditor(QtWidgets.QDialog):
    def __init__(self, parent, result):
        super().__init__(parent)
        self.setWindowTitle("Change Path")
        with QtShortCuts.QVBoxLayout(self) as layout:
            QtWidgets.QLabel("Change path where the images are stored.").addToLayout()
            self.label = QtShortCuts.SuperQLabel(
                f"The current path is {result.template}.").addToLayout()
            self.label.setWordWrap(True)
            self.input_pack = QtShortCuts.QInputBool(None, "pack files", False)
            self.input_save = QtShortCuts.QInputBool(None, "save", False)
            self.input_folder = QtShortCuts.QInputFolder(None, "", Path(result.template), allow_edit=True)
            if result.stack_reference:
                self.input_folder2 = QtShortCuts.QInputFolder(None, "", Path(result.stack_reference.template),
                                                              allow_edit=True)
            else:
                self.input_folder2 = None
            with QtShortCuts.QHBoxLayout():
                self.button_addList0 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                self.button_addList0 = QtShortCuts.QPushButton(None, "ok", self.accept)


class PathChanger:
    def __init__(self, old_template, new_template):
        old_template = str(old_template)
        new_template = str(new_template)

        old_template = re.sub(r"\{c(:[^}]*)?\}", r"{c}", old_template)
        old_template = re.sub(r"\[z\]$", "[{z}]", old_template)
        old_template_re = re.sub(r"\\\{c(:[^}]*)?\\\}", r"(?P<c>.*)",
                                 re.escape(old_template).replace("\\{t\\}", r"(?P<t>.*)").replace("\\{z\\}",
                                                                                                  r"(?P<z>.*)"))
        self.old_template_re = re.compile(old_template_re)

        self.new_template = new_template
        self.new_template_format = re.sub(r"\{c(:[^}]*)?\}", r"{c}", new_template)
        self.new_template_format = re.sub(r"\[z\]$", "[{z}]", self.new_template_format)

    def change_path(self, path):
        path_type = isinstance(path, Path)
        path = str(path)
        match = self.old_template_re.match(path)
        if match is None:
            raise ValueError(f"Path {path} does not fit template {self.old_template_re}")
        new = self.new_template_format.format(**match.groupdict())
        #print("change_path From", path)
        #print("change_path To  ", new)
        if path_type:
            return Path(new)
        return new

    def stack_change(self, stack: "Stack"):
        for index in range(len(stack.image_filenames)):
            for index2 in range(len(stack.image_filenames[index])):
                stack.image_filenames[index][index2] = self.change_path(stack.image_filenames[index][index2])
        stack.template = self.change_path(stack.template)
