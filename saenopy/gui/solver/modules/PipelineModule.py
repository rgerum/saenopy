import qtawesome as qta
from qtpy import QtCore, QtWidgets
from saenopy import Result
from typing import Tuple, List
import traceback


class ParameterMapping:
    params_name: str = None
    parameter_dict: dict = None

    result: Result = None

    def __init__(self, params_name: str = None, parameter_dict: dict=None):
        self.params_name = params_name
        self.parameter_dict = parameter_dict
        for name, widget in self.parameter_dict.items():
            widget.valueChanged.connect(lambda x, name=name: self.setParameter(name, x))

        self.setResult(None)

    def setParameter(self, name: str, value):
        if self.result is not None:
            getattr(self.result, self.params_name + "_tmp")[name] = value

    def ensure_tmp_params_initialized(self, result):
        if self.params_name is None:
            return
        # if the results instance does not have the parameter dictionary yet, create it
        if getattr(result, self.params_name + "_tmp", None) is None:
            setattr(result, self.params_name + "_tmp", {})

        # set the widgets to the value if the value exits
        params = getattr(result, self.params_name)
        params_tmp = getattr(result, self.params_name + "_tmp")
        # iterate over the parameters
        for name, widget in self.parameter_dict.items():
            if name not in params_tmp:
                if params is not None and name in params:
                    params_tmp[name] = params[name]
                else:
                    params_tmp[name] = widget.value()

    def setDisabled(self, disabled):
        # disable all the widgets
        for name, widget in self.parameter_dict.items():
            widget.setDisabled(disabled)

    def setResult(self, result: Result):
        """ set a new active result object """
        self.result = result

        # if a result file is given
        if result is not None:
            self.ensure_tmp_params_initialized(result)
            params_tmp = getattr(result, self.params_name + "_tmp")
            # iterate over the parameters
            for name, widget in self.parameter_dict.items():
                # set the widgets to the value if the value exits
                widget.setValue(params_tmp[name])


class PipelineModule(QtWidgets.QWidget):
    processing_finished = QtCore.Signal()
    processing_progress = QtCore.Signal(tuple)
    processing_state_changed = QtCore.Signal(object)
    processing_error = QtCore.Signal(str)
    result: Result = None
    tab: QtWidgets.QTabWidget = None

    parameter_mappings: List[ParameterMapping] = None
    params_name: None

    def __init__(self, parent: "BatchEvaluate", layout):
        super().__init__()
        self.parameter_mappings = []
        self.params_name = None

        if layout is not None:
            layout.addWidget(self)
        if parent is None:
            return
        self.parent = parent
        self.settings = self.parent.settings

        self.processing_finished.connect(self.finished_process)
        self.processing_error.connect(self.errored_process)
        self.processing_state_changed.connect(self.state_changed)

        self.parent.result_changed.connect(self.resultChanged)
        self.parent.set_current_result.connect(self.setResult)
        self.parent.tab_changed.connect(self.tabChanged)

        self.processing_progress.connect(self.parent.progress)

    def setParameterMapping(self, params_name: str = None, parameter_dict: dict=None):
        self.params_name = params_name
        if params_name is None:
            return
        self.parameter_mappings.append(ParameterMapping(params_name, parameter_dict))

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

    def check_available(self, result: Result) -> bool:
        return False

    def check_evaluated(self, result: Result) -> bool:
        return False

    def resultChanged(self, result: Result):
        """ called when the contents of result changed. Only update view if it is the currently displayed one. """
        if result is self.result:
            if self.tab is not None:
                for i in range(self.parent.tabs.count()):
                    if self.parent.tabs.widget(i) == self.tab.parent():
                       self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))
            if self.current_tab_selected is True:
                self.update_display()
            self.state_changed(result)

    def state_changed(self, result: Result):
        if result is self.result and getattr(self, "group", None) is not None:
            state = getattr(result, self.params_name + "_state", "")
            if state == "scheduled":
                self.group.label.setIcon(qta.icon("fa5s.hourglass-start", options=[dict(color="gray")]))
                self.group.label.setToolTip("scheduled")
            elif state == "running":
                self.group.label.setIcon(qta.icon("fa5s.hourglass-half", options=[dict(color="orange")]))
                self.group.label.setToolTip("running")
            elif state == "finished":
                self.group.label.setIcon(qta.icon("fa5s.hourglass-end", options=[dict(color="green")]))
                self.group.label.setToolTip("finished")
            elif state == "failed":
                self.group.label.setIcon(qta.icon("fa5s.times", options=[dict(color="red")]))
                self.group.label.setToolTip("failed")
            else:
                self.group.label.setIcon(qta.icon("fa5.circle", options=[dict(color="gray")]))
                self.group.label.setToolTip("")

            if state == "scheduled" or state == "running":
                # if not disable all the widgets
                for mapping in self.parameter_mappings:
                    mapping.setDisabled(True)
                if getattr(self, "input_button", None):
                    self.input_button.setEnabled(False)
            else:
                # if not disable all the widgets
                for mapping in self.parameter_mappings:
                    mapping.setDisabled(False)
                if getattr(self, "input_button", None):
                    self.input_button.setEnabled(self.check_available(result))
            #if getattr(self, "input_button", None):
            #    self.input_button.setEnabled(self.check_available(result))

    def setResult(self, result: Result):
        """ set a new active result object """
        #if result == self.result:
        #    return
        self.current_result_plotted = False
        self.result = result

        for mapping in self.parameter_mappings:
            mapping.setResult(result)

        if result is not None:
            self.t_slider.setRange(0, len(result.stacks) - 2)

        self.state_changed(result)
        if self.tab is not None:
            for i in range(self.parent.tabs.count()):
                if self.parent.tabs.widget(i) == self.tab.parent():
                    self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))

        # check if the results instance can be evaluated currently with this module
        #if self.check_available(result) is False:
        if getattr(self, "input_button", None):
            self.input_button.setEnabled(self.check_available(result))
        if result is None or \
                (self.params_name and (getattr(result, self.params_name + "_state", "") == "scheduled"
                                       or getattr(result, self.params_name + "_state", "") == "running")):
            # if not disable all the widgets
            for mapping in self.parameter_mappings:
                mapping.setDisabled(True)
            if getattr(self, "input_button", None):
                self.input_button.setEnabled(False)
        else:
            # if not disable all the widgets
            for mapping in self.parameter_mappings:
                mapping.setDisabled(False)
            self.valueChanged()
        if self.current_tab_selected is True:
            self.update_display()

    def update_display(self):
        pass

    def valueChanged(self):
        pass

    def start_process(self, x=None, result=None):
        if result is None:
            result = self.result
        if result is None:
            return
        if getattr(result, self.params_name + "_state", "") == "scheduled" or \
            getattr(result, self.params_name + "_state", "") == "running":
            return

        params = {}
        for mapping in self.parameter_mappings:
            mapping.ensure_tmp_params_initialized(result)
            params[mapping.params_name] = getattr(result, mapping.params_name + "_tmp")
        setattr(result, self.params_name + "_state", "scheduled")
        self.processing_state_changed.emit(result)
        return self.parent.addTask(self.process_thread, result, params, "xx")

    def process_thread(self, result: Result, params: dict):
        #params = getattr(result, self.params_name + "_tmp")
        self.parent.progressbar.setRange(0, 0)
        setattr(result, self.params_name + "_state", "running")
        self.processing_state_changed.emit(result)
        try:
            self.process(result, **params)
            # store the parameters that have been used for evaluation
            for mapping in self.parameter_mappings:
                setattr(result, mapping.params_name, params[mapping.params_name].copy())
            result.save()
            setattr(result, self.params_name + "_state", "finished")
            self.parent.result_changed.emit(result)
            self.processing_finished.emit()
        except Exception as err:
            traceback.print_exc()
            setattr(result, self.params_name + "_state", "failed")
            self.processing_state_changed.emit(result)
            self.processing_error.emit(str(err))

    def process(self, result: Result, params: dict):
        pass

    def finished_process(self):
        self.parent.progressbar.setRange(0, 1)

    def errored_process(self, text: str):
        QtWidgets.QMessageBox.critical(self, "Deformation Detector", text)
        self.parent.progressbar.setRange(0, 1)

    def get_code(self) -> Tuple[str, str]:
        return "", ""
