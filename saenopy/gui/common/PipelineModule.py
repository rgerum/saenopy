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


from enum import Enum

class StateEnum(str, Enum):
    idle = ""
    scheduled = "scheduled"
    running = "running"
    finished = "finished"
    failed = "failed"


class PipelineModule(QtWidgets.QWidget):
    processing_finished = QtCore.Signal()
    processing_progress = QtCore.Signal(tuple)
    processing_state_changed = QtCore.Signal(object)
    processing_error = QtCore.Signal(str)
    result: Result = None
    tab: QtWidgets.QTabWidget = None

    parameter_mappings: List[ParameterMapping] = None
    params_name: None

    pipeline_allow_cancel = False

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
        if getattr(self, "input_button", None) is not None:
            self.input_button.setEnabled(False)
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
            state = self.get_result_state(result)
            if state == StateEnum.scheduled:
                self.group.label.setIcon(qta.icon("fa5s.hourglass-start", options=[dict(color="gray")]))
                self.group.label.setToolTip("scheduled")
                if self.pipeline_allow_cancel:
                    my_state, count, max = self.check_status(result)
                    self.input_button.setText(f"cancel {count}/{max}")
                self.input_button.setEnabled(False)

            elif state == StateEnum.running:
                self.group.label.setIcon(qta.icon("fa5s.hourglass-half", options=[dict(color="orange")]))
                self.group.label.setToolTip("running")
                if self.pipeline_allow_cancel:
                    my_state, count, max = self.check_status(result)
                    self.input_button.setText(f"cancel {count}/{max}")
                    self.input_button.setEnabled(True)
                else:
                    self.input_button.setEnabled(False)

            elif state == StateEnum.finished:
                self.group.label.setIcon(qta.icon("fa5s.hourglass-end", options=[dict(color="green")]))
                self.group.label.setToolTip("finished")
                self.input_button.setText(f"finished")
                self.input_button.setEnabled(False)

            elif state == StateEnum.failed:
                self.group.label.setIcon(qta.icon("fa5s.times", options=[dict(color="red")]))
                self.group.label.setToolTip("failed")
                self.input_button.setEnabled(False)
            else:
                self.group.label.setIcon(qta.icon("fa5.circle", options=[dict(color="gray")]))
                self.group.label.setToolTip("")

                if self.pipeline_allow_cancel is True and self.check_available(result):
                    my_state, count, max = self.check_status(result)
                    if my_state == "progress":
                        if count == 0:
                            self.input_button.setText(f"detect deformation")
                        else:
                            self.input_button.setText(f"continue {count}/{max}")
                        self.input_button.setEnabled(True)
                    else:
                        self.input_button.setText(f"done")
                        self.input_button.setEnabled(False)
                else:
                    self.input_button.setEnabled(self.check_available(result))

            if state == StateEnum.scheduled or state == StateEnum.running:
                # if not disable all the widgets
                for mapping in self.parameter_mappings:
                    mapping.setDisabled(True)
            else:
                # if not disable all the widgets
                for mapping in self.parameter_mappings:
                    mapping.setDisabled(False)
            #if getattr(self, "input_button", None):
            #    self.input_button.setEnabled(self.check_available(result))

    def setResult(self, result: Result):
        """ set a new active result object """
        self.current_result_plotted = False
        self.result = result

        for mapping in self.parameter_mappings:
            mapping.setResult(result)

        if result is not None and getattr(self, "t_slider", None) is not None:
            data = result.get_data_structure()
            self.t_slider.setRange(0, data["time_point_count"] - 1)

        self.state_changed(result)
        if self.tab is not None:
            for i in range(self.parent.tabs.count()):
                if self.parent.tabs.widget(i) == self.tab.parent():
                    self.parent.tabs.setTabEnabled(i, self.check_evaluated(result))

        if result is None or \
                (self.params_name and (self.get_result_state(result) == "scheduled"
                                       or self.get_result_state(result) == "running")):
            # if not disable all the widgets
            for mapping in self.parameter_mappings:
                mapping.setDisabled(True)
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

    def get_result_state(self, result) -> StateEnum:
        return getattr(result, self.params_name + "_state", "")

    def set_result_state(self, result: Result, state: StateEnum):
        setattr(result, self.params_name + "_state", state)

    def start_process(self, x=None, result=None):
        if result is None:
            result = self.result
        if result is None:
            return

        state = self.get_result_state(result)

        if state == StateEnum.scheduled:
            return

        if self.pipeline_allow_cancel is True:
            if state == StateEnum.running:
                return self.cancel_process()

        if state == StateEnum.running:
            return

        params = {}
        for mapping in self.parameter_mappings:
            mapping.ensure_tmp_params_initialized(result)
            params[mapping.params_name] = getattr(result, mapping.params_name + "_tmp")
        self.set_result_state(result, StateEnum.scheduled)
        self.processing_state_changed.emit(result)
        return self.parent.addTask(self.process_thread, result, params, "xx")

    def process_thread(self, result: Result, params: dict):
        #params = getattr(result, self.params_name + "_tmp")
        self.parent.progressbar.setRange(0, 0)
        self.set_result_state(result, StateEnum.running)
        self.processing_state_changed.emit(result)
        try:
            res = self.process(result, **params)
            if res == "Terminated":
                self.set_result_state(result, StateEnum.idle)
                self.parent.result_changed.emit(result)
                self.processing_finished.emit()
                return
            # store the parameters that have been used for evaluation
            for mapping in self.parameter_mappings:
                setattr(result, mapping.params_name, params[mapping.params_name].copy())
            result.save()
            self.set_result_state(result, StateEnum.finished)
            self.parent.result_changed.emit(result)
            self.processing_finished.emit()
        except Exception as err:
            traceback.print_exc()
            self.set_result_state(result, StateEnum.failed)
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
