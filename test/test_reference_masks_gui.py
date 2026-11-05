from pathlib import Path
import numpy as np
import pytest
import tifffile
from test_orientation_workflow import window

@pytest.mark.parametrize('mode',['point','line'])
@pytest.mark.parametrize('source',['whole_image','same_image','separate_image'])
def test_reference_mask_import_preview_analysis_reload(window,tmp_path,monkeypatch,mode,source):
    from qtpy import QtCore,QtTest,QtWidgets
    from saenopy.gui.orientation.modules.AddFilesDialog import AddFilesDialog
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    from saenopy.gui.orientation.modules.SegmentationPreview import calculate_preview
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    yy,xx=np.indices((128,160))
    image=(20000+18000*np.cos(xx*.3+yy*.12)).astype(np.uint16)
    fiber=tmp_path/'fiber.tif'
    other=tmp_path/'mask.tif'
    tifffile.imwrite(fiber,image)
    tifffile.imwrite(other,(((xx-80)**2+(yy-64)**2<40**2)*50000).astype(np.uint16))
    def accept(dialog):
        dialog.show()
        dialog.analysis_mode_selector.buttons['point'].click()
        dialog.reference_type.buttons[mode].click()
        dialog.mask_source.buttons[source].click()
        dialog.fiberText.setValue(str(fiber))
        dialog.cellText.setValue(str(other))
        dialog.outputText.setValue(str(tmp_path/'output'))
        assert dialog.cellText.isVisible()==(source=='separate_image')
        assert dialog.button_addList1.isEnabled()
        dialog.accept_new()
        return 1
    monkeypatch.setattr(AddFilesDialog,'exec',accept)
    batch=window.deformations
    batch.add_measurement()
    result=batch.data[-1][2]
    assert result.mask_source==source
    assert result.uses_mask==(source!='whole_image')
    assert batch.sub_module_segmentation.isHidden()==(source=='whole_image')
    p=dict(edge=10,sigma_tensor=4.,sigma_tensor_type='pixel',sigma_first_blur=.5,
           reference_x=80.,reference_y=64.,line_x1=30.,line_y1=90.,line_x2=130.,line_y2=40.)
    seg=dict(mask_sigma=3.,mask_threshold=.5,mask_invert=False)
    result.orientation_parameters_tmp=p.copy()
    if result.uses_mask:
        preview=calculate_preview(result,seg,p)
        result.mask_preview=preview['mask']
        result.mask_preview_edge=preview['edge']
        batch.tab1.update_display()
        assert batch.tab1.mask_overlay.isVisible()
        assert batch.tab1.mask_overlay.pixmap().size()==QtCore.QSize(160,128)
        assert batch.tab1.pixmap.pixmap().size()==QtCore.QSize(160,128)
    analyze_orientation(result,p,seg)
    loaded=ResultOrientation.load(result.output)
    assert loaded.mask_source==source
    assert loaded.orientation_map.shape==(108,140)
    if source == 'separate_image':
        assert loaded.validity_mask.any()
        assert not loaded.validity_mask.all()
    assert np.isnan(loaded.orientation_map[~loaded.validity_mask]).all()
    batch.set_current_result.emit(loaded)
    batch.tabs.setCurrentWidget(batch.tab1.tab.parent())
    QtWidgets.QApplication.processEvents()
    # References keep original-image coordinates, even with a cropped mask.
    assert batch.tab1.pixmap.pixmap().size()==QtCore.QSize(160,128)
    if mode=='point':
        assert batch.tab1.point_marker.pos()==QtCore.QPointF(80.5,64.5)
        tab=batch.tab1
        tab.choose_point.click()
        position=tab.label.mapFromScene(tab.label.origin.mapToScene(QtCore.QPointF(60.5,40.5)))
        QtTest.QTest.mouseClick(tab.label.viewport(),QtCore.Qt.LeftButton,pos=position)
        assert loaded.orientation_parameters_tmp['reference_x']==60
        assert loaded.orientation_parameters['reference_x']==80
    stats=PlottingWindow.load_file(None,result.output)
    assert stats.resulting_data['Alignment '+mode][0]==pytest.approx(np.nanmean(loaded.orientation_map))
    assert stats.resulting_data['Mean Coherency'][0]==pytest.approx(np.nanmean(loaded.coherence_map))
