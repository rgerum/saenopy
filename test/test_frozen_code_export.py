"""Preserve exported code when packaging functions at different nesting depths."""
from saenopy.add_source import add_source_code_to_decorator
from saenopy.gui.common.code_export import get_code


def test_packaged_sources_follow_file_order_not_ast_depth(tmp_path):
    path = tmp_path / "exports.py"
    path.write_text('''
def export_as_string(function):
    return function

def exports():
    if True:
        if True:
            @export_as_string
            def earlier(value1):
                result = value1 + 1
    if True:
        @export_as_string
        def later(value2):
            result = value2 + 2
    return earlier, later
''', encoding="utf-8")
    add_source_code_to_decorator(tmp_path)
    namespace = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    earlier, later = namespace["exports"]()
    for function, replacements, expected in (
            (earlier, {"value1": 10}, 11), (later, {"value2": 20}, 22)):
        output = {}
        exec(get_code(function, replacements), output)
        assert output["result"] == expected
