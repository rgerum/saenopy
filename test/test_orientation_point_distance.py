from pathlib import Path
import numpy as np
import pytest
import tifffile
from test_orientation_workflow import window


def measurement(tmp_path, name, width=10, pixel_size=.5):
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    yy,xx=np.indices((96,120))
    image=tmp_path/(name+'.tif')
    tifffile.imwrite(image,(30000+25000*np.cos(xx*.3+yy*.1)).astype(np.uint16))
    result=ResultOrientation(output=str(tmp_path/(name+'.saenopyOrientation')),
        image_fiber=image.name,image_cell=None,pixel_size=pixel_size,analysis_mode='point_orientation')
    params=dict(sigma_tensor=3.,sigma_tensor_type='pixel',sigma_first_blur=.5,edge=5,
                reference_x=60.,reference_y=48.,shell_width=width,shell_width_type='pixel')
    analyze_orientation(result,params)
    return result


def test_api_gui_archive_and_legacy_parity(window,tmp_path,monkeypatch):
    from CompactionAnalyzer import analyze_image
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.analyze.PlottingWindow import PlottingWindow
    result=measurement(tmp_path,'point',width=7.5)
    expected=analyze_image(result.get_absolute_path_fiber(),mode='point_orientation',
        reference_point=(60,48),edge=5,sigma_tensor=3,scale=.5,shell_width=7.5,outputs='distance_profile')
    from saenopy.gui.orientation.modules.point_distance import PROFILE_COLUMNS
    loaded=ResultOrientation.load(result.output)
    for key,column in PROFILE_COLUMNS.items():
        np.testing.assert_allclose([row[column] for row in loaded.results_distance],expected[key],atol=1e-12,equal_nan=True)
    # Old archives have the same maps but no profile. Derive it without the images/tensor.
    loaded.results_distance=[]
    loaded.save()
    before=Path(result.output).read_bytes()
    monkeypatch.setattr('CompactionAnalyzer.analyze_image',lambda *a,**k:pytest.fail('No reanalysis on load'))
    legacy=PlottingWindow.load_file(None,result.output)
    np.testing.assert_allclose(legacy.resulting_distance_data['Point alignment (shell)'],expected['mean_alignment_point'],atol=1e-12)
    assert Path(result.output).read_bytes()==before
    assert len(legacy.resulting_data)==1
    module=window.deformations.sub_module_deformation
    window.deformations.add_data(loaded)
    assert not module.shell_width.isHidden()
    assert not module.shell_width_type.isHidden()


def test_group_single_units_export_and_global_bar(window,tmp_path,monkeypatch):
    first=measurement(tmp_path,'one')
    second=measurement(tmp_path,'two')
    plotter=window.plotting_window
    plotter.add_files([first.output,second.output])
    keys=plotter.point_distance_metrics
    assert keys <= set(plotter.update_metric_availability())
    for metric,limits in [('point alignment over distance',(-1,1)),
                          ('point angular deviation over distance',(0,90)),
                          ('point coherency over distance',(0,1))]:
        plotter.type.setValue(metric,send_signal=True)
        assert plotter.button_run.isEnabled() and plotter.button_run2.isEnabled()
        assert not plotter.button_run3.isEnabled() and not plotter.agg.isEnabled()
        assert plotter.export_data is not None
        assert plotter.canvas.axes.get_ylim()==limits
        curve=plotter.canvas.axes.lines[0]
        rows=plotter.data_folders[0][2][0][2].resulting_distance_data
        _,column,_=plotter.metrics[metric]
        np.testing.assert_allclose(curve.get_ydata(),rows[column],atol=1e-12)
        np.testing.assert_allclose(curve.get_xdata(),rows['Shell_mid (µm)'])
        plotter.distance_unit.setValue('pixel',send_signal=True)
        np.testing.assert_allclose(plotter.canvas.axes.lines[0].get_xdata(),rows['Shell_mid (px)'])
        plotter.list2.setCurrentRow(0)
        plotter.run2()
        assert plotter.export_data[1][0].filename.nunique()==1
        plotter.plot_groups()
        assert plotter.export_data[1][0].filename.nunique()==2
        # Execute the actual exported Python plot, using the CSV round-trip
        # performed by the export dialog, and compare its curves and limits.
        import io
        import pandas as pd
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)
        code, (data, columns) = plotter.export_data
        expected_x=plotter.canvas.axes.lines[0].get_xdata().copy()
        expected_y=plotter.canvas.axes.lines[0].get_ydata().copy()
        figure=plt.figure()
        exec(code, {'plt':plt, 'res':pd.read_csv(io.StringIO(data.to_csv(columns=columns)))})
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(),expected_x)
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(),expected_y)
        assert figure.axes[0].get_ylim()==limits
        plt.close(figure)
        plotter.distance_unit.setValue('µm',send_signal=True)
    # Adding shells must not change one-value-per-image mean alignment.
    plotter.type.setValue('point alignment',send_signal=True)
    values=plotter.export_data[1][0]
    assert len(values)==2
    expected=np.nanmean(first.orientation_map)
    np.testing.assert_allclose(values['Alignment point'],expected,atol=1e-12)
    assert plotter.button_run3.isEnabled()
    assert plotter.distance_unit.isHidden()


def test_incompatible_widths_and_units_are_not_combined(window,tmp_path):
    a=measurement(tmp_path,'a',pixel_size=.5)
    b=measurement(tmp_path,'b',pixel_size=1.)
    plotter=window.plotting_window
    plotter.add_files([a.output,b.output])
    plotter.type.setValue('point alignment over distance',send_signal=True)
    assert plotter.export_data is None
    assert 'matching shell widths' in plotter.metric_note.text()
    plotter.distance_unit.setValue('pixel',send_signal=True)
    assert plotter.export_data is not None


def test_group_means_and_sem_weight_images_not_pixels(window,tmp_path):
    first=measurement(tmp_path,'first')
    second=measurement(tmp_path,'second')
    for result,mean,count in [(first,.2,10000),(second,.8,10)]:
        for row in result.results_distance:
            row['Point alignment (shell)']=mean
            row['Valid pixels']=count
        result.save()
    # The first image has no data in one ring, so only the second contributes.
    first.results_distance[2]['Point alignment (shell)']=np.nan
    first.results_distance[2]['Valid pixels']=0
    first.save()
    plotter=window.plotting_window
    plotter.add_files([first.output,second.output])
    plotter.type.setValue('point alignment over distance',send_signal=True)
    curve=plotter.canvas.axes.lines[0]
    expected=np.full(len(first.results_distance),.5)
    expected[2]=.8
    np.testing.assert_allclose(curve.get_ydata(),expected)
    # SEM for [.2, .8] is .3; the band runs from .2 to .8, irrespective of counts.
    vertices=np.concatenate([path.vertices for path in plotter.canvas.axes.collections[0].get_paths()])
    np.testing.assert_allclose([vertices[:,1].min(),vertices[:,1].max()],[.2,.8])
    assert 'n=1–2' in curve.get_label()


def test_empty_rings_remain_gaps_in_grouped_plot(window,tmp_path):
    result=measurement(tmp_path,'gaps')
    for row in result.results_distance[2:4]:
        row['Point alignment (shell)']=np.nan
        row['Valid pixels']=0
    result.save()
    plotter=window.plotting_window
    plotter.add_files([result.output])
    plotter.type.setValue('point alignment over distance',send_signal=True)
    assert np.isnan(plotter.canvas.axes.lines[0].get_ydata()[2:4]).all()
