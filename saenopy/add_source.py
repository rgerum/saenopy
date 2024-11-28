import ast
import os
from pathlib import Path

def add_source_code_to_decorator(folder_path):
    """
    Go through all Python files in a folder, find functions decorated with @export_as_string,
    and add an attribute to those functions containing their source code.
    """
    for py_file in Path(folder_path).rglob("*.py"):  # Recursively find all .py files
        #if not str(py_file).endswith("DeformationDetector.py"):
        #    continue
        #print(py_file)
        with open(py_file, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Parse the Python file into an AST
        tree = ast.parse(source_code)
        lines = source_code.splitlines()
        modified = False  # Track if modifications are made
        line_shift = 0  # Track cumulative line number shifts

        # Traverse all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):  # Process function definitions
                # Check if the function has @export_as_string
                if node.decorator_list and any(
                        isinstance(decorator, ast.Name) and decorator.id == "export_as_string"
                        for decorator in node.decorator_list
                ):
                    # Extract the function's source code, including trailing comments
                    func_source_start = node.lineno - 1 + line_shift
                    func_source_end = node.end_lineno + line_shift

                    # Extend the end to include trailing comments
                    while func_source_end < len(lines) and lines[func_source_end].strip().startswith("#"):
                        func_source_end += 1

                    # Extract the full function source
                    func_source_lines = lines[func_source_start:func_source_end]
                    func_source = "\n".join(func_source_lines)

                    # Check if the _source_code attribute is already added
                    attribute_name = f"{node.name}._source_code = '''\n{func_source}\n'''"
                    if attribute_name not in "\n".join(lines):
                        # Determine the indentation level of the function
                        func_indent = len(func_source_lines[0]) - len(func_source_lines[0].lstrip())
                        indent = " " * func_indent

                        # Insert the _source_code attribute after the function and trailing comments
                        insert_position = func_source_end
                        lines.insert(insert_position, f"{indent}{attribute_name}")
                        line_shift += 1  # Increment the shift counter for future insertions
                        modified = True

        # If the file was modified, write it back
        if modified:
            with open(py_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"Updated {py_file}")

if __name__ == "__main__":
    folder_path = Path(__file__).parent
    if os.path.exists(folder_path):
        add_source_code_to_decorator(folder_path)
        print("done")
    else:
        print("Invalid folder path.")
