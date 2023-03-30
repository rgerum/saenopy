from pathlib import Path
import shutil


class MockDir:
    def __init__(self, structure, callback=None):
        self.structure = structure
        self.callback = callback

    def __enter__(self):
        def mock_dir(structure, parent=None):
            if parent is None:
                parent = Path(".")
            if isinstance(structure, list):
                for file in structure:
                    if self.callback:
                        self.callback(parent / file)
                    else:
                        (parent / file).touch(exist_ok=True)
            else:
                for key in structure:
                    folder = parent / key
                    folder.mkdir(exist_ok=True)
                    mock_dir(structure[key], folder)

        mock_dir(self.structure)

    def __exit__(self, exc_type, exc_val, exc_tb):
        def remove_mock(structure, parent=None):
            if parent is None:
                parent = Path(".")
            if isinstance(structure, list):
                for file in structure:
                    (parent / file).unlink(missing_ok=True)
            else:
                for key in structure:
                    folder = parent / key
                    remove_mock(structure[key], folder)
                    #folder.rmdir()
                    shutil.rmtree(folder, ignore_errors=False, onerror=None)
        remove_mock(self.structure)
