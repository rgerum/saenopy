"""Keep GUI integration tests independent of persistent application settings."""
import pytest


@pytest.fixture
def isolated_qt_settings(tmp_path, monkeypatch):
    from qtpy import QtCore

    original_settings = QtCore.QSettings
    # The organization/application constructor still uses native settings even
    # after setDefaultFormat(IniFormat); explicitly select a temporary file.
    monkeypatch.setattr(QtCore, "QSettings", lambda *args, **kwargs: original_settings(
        str(tmp_path / "settings.ini"), original_settings.IniFormat))
