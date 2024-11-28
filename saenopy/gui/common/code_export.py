import inspect


def export_as_string(func):
    try:
        func._source_code = inspect.getsource(func)
    except OSError:  # this error occurs in PyInstaller executable and is fixed by inlining the source code there
        pass
    return func


def get_code(code, data):
    #code_lines = inspect.getsource(code).split("\n")[1:]
    code_lines = code._source_code.split("\n")[2:]

    indent = len(code_lines[0]) - len(code_lines[0].lstrip())
    code = "\n".join(line[indent:] for line in code_lines)

    for key, value in data.items():
        if isinstance(value, str):
            if "\\" in value:
                code = code.replace(key, "r'" + value + "'")
            else:
                code = code.replace(key, "'" + value + "'")
        else:
            code = code.replace(key, str(value))
    return code
