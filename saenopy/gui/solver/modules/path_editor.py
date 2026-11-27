from qtpy import QtWidgets
from pathlib import Path
import re
import numpy as np

from saenopy.gui.common import QtShortCuts
from saenopy.result_file import Result


def start_path_change(parent, result: Result):

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

    for source, widget in zip(result.image_sources, path_editor.input_sources):
        path_changer = PathChanger(source.template, widget.value())
        for stack in source.stacks:
            path_changer.stack_change(stack)
        source.template = path_changer.change_path(source.template)

    if path_editor.input_pack.value():
        for stack in result.stacks:
            stack.pack_files()
        if result.stack_reference:
            result.stack_reference.pack_files()
        for source in result.image_sources:
            for stack in source.stacks:
                stack.pack_files()

    if path_editor.input_save.value():
        result.save()
        #print("saved")
        return


class PathEditor(QtWidgets.QDialog):
    def __init__(self, parent, result):
        super().__init__(parent)
        self.setWindowTitle("Change Path")

        def folder_input(name, template):
            widget = QtShortCuts.QInputFolder(None, name, Path(template), allow_edit=True)
            widget.filename_checker = lambda folder: str(Path(folder) / Path(widget.value()).name)
            return widget

        with QtShortCuts.QVBoxLayout(self) as layout:
            QtWidgets.QLabel("Change path where the images are stored.").addToLayout()
            self.label = QtShortCuts.SuperQLabel(
                f"The current path is {result.template}.").addToLayout()
            self.label.setWordWrap(True)
            self.input_pack = QtShortCuts.QInputBool(None, "pack files", False)
            self.input_save = QtShortCuts.QInputBool(None, "save", False)
            self.input_folder = folder_input("", result.template)
            if result.stack_reference:
                self.input_folder2 = folder_input("", result.stack_reference.template)
            else:
                self.input_folder2 = None
            self.input_sources = []
            for source in result.image_sources:
                if PathChanger.leica_suffix.search(str(source.template)):
                    widget = QtShortCuts.QInputFilename(
                        None, source.name, str(source.template), existing=True,
                        file_type="Leica (*.lif)", allow_edit=True)
                else:
                    widget = folder_input(source.name, source.template)
                self.input_sources.append(widget)
            with QtShortCuts.QHBoxLayout():
                self.button_addList0 = QtShortCuts.QPushButton(None, "cancel", self.reject)
                self.button_addList0 = QtShortCuts.QPushButton(None, "ok", self.accept)


class PathChanger:
    leica_suffix = re.compile(r"\{f:\d+\}\{c:\d+\}(?:\{t:\d+\})?\.lif$")

    def __init__(self, old_template, new_template):
        old_template = str(old_template)
        new_template = str(new_template)
        self.leica_paths = None
        suffix = self.leica_suffix.search(old_template)
        if suffix is not None:
            if not new_template.endswith(".lif"):
                raise ValueError("Select a Leica .lif file for this source.")
            new_suffix = self.leica_suffix.search(new_template)
            self.leica_paths = (old_template[:suffix.start()],
                                new_template[:new_suffix.start()] if new_suffix else new_template[:-4])
            return

        old_template = re.sub(r"\{c(:[^}]*)?\}", r"{c}", old_template)
        old_template = re.sub(r"\[([zt])\]$", r"[{\1}]", old_template)
        old_template_re = re.sub(r"\\\{c(:[^}]*)?\\\}", r"(?P<c>.*)",
                                 re.escape(old_template).replace("\\{t\\}", r"(?P<t>.*)").replace("\\{z\\}",
                                                                                                  r"(?P<z>.*)"))
        self.old_template_re = re.compile(old_template_re)

        self.new_template = new_template
        self.new_template_format = re.sub(r"\{c(:[^}]*)?\}", r"{c}", new_template)
        self.new_template_format = re.sub(r"\[([zt])\]$", r"[{\1}]", self.new_template_format)

    def change_path(self, path):
        path_type = isinstance(path, Path)
        path = str(path)
        if self.leica_paths is not None:
            suffix = self.leica_suffix.search(path)
            if suffix is None or Path(path[:suffix.start()]) != Path(self.leica_paths[0]):
                raise ValueError(f"Path {path} does not fit the selected Leica source.")
            # Relocate the container, retaining each stack's folder/channel/time.
            new = self.leica_paths[1] + suffix.group()
        else:
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
        if self.leica_paths is not None:
            stack.template = self.change_path(stack.template)
            stack.leica_filename = self.leica_paths[1]
            stack._leica_file = None
            return
        # Rebuild the string array so a longer destination is not truncated to
        # the old NumPy fixed-width string dtype.
        stack.image_filenames = np.asarray([
            [self.change_path(filename) for filename in row]
            for row in stack.image_filenames])
        stack.template = self.change_path(stack.template)
