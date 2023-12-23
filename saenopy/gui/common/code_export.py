import inspect


def get_code(code, data):
    code_lines = inspect.getsource(code).split("\n")[1:]
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
