import re

def parse_pyx(pyx_code):
    # Replace component tags with function calls
    py_code = pyx_code

    # Handle component tags
    py_code = re.sub(r'<(\w+)(.*?)>(.*?)</\1>', r'create_element("\1", {\2}, \3)', py_code, flags=re.DOTALL)

    # Handle self-closing tags
    py_code = re.sub(r'<(\w+)(.*?)\s*/>', r'create_element("\1", {\2})', py_code)

    # Handle props
    py_code = re.sub(r'(\w+)=(\{.*?\})', r'"\1": \2', py_code)

    # Handle lambda functions
    py_code = re.sub(r'lambda\s*:\s*', r'lambda x: ', py_code)

    # Handle map function
    py_code = re.sub(r'map\((.*?)\)', r'list(map(\1))', py_code)

    # Handle conditional rendering
    py_code = re.sub(r'\{(.*?)\s+and\s+\((.*?)\)\}', r'(\1 if \1 else \2)', py_code)

    return py_code

def convert_pyx_to_py(pyx_file, py_file):
    with open(pyx_file, 'r') as f:
        pyx_code = f.read()

    # Parse the pyx code to convert it to Python code
    py_code = parse_pyx(pyx_code)

    # Write the converted Python code to a new file
    with open(py_file, 'w') as f:
        f.write(py_code)

# Example usage
if __name__ == "__main__":
    convert_pyx_to_py('app.pyx.py', 'output.py')
