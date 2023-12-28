import ctypes


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
