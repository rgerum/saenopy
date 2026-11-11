"""Copyable direct API examples for the existing measurement help panel."""
from html import escape


def analysis_api_example(mode):
    options = {
        "full": ('    mask_image="cell.tif",\n', "mean_alignment_cell"),
        "axis_orientation_only": ("", "mean_alignment_x"),
        "axis_orientation_masked": ("", "mean_alignment_x"),
        "point_orientation": ('    reference_point=(320, 240),\n', "mean_alignment_point"),
        "line_orientation": ('    reference_line=(\n        (80, 240),\n        (560, 120),\n    ),\n', "mean_alignment_line"),
    }
    arguments, output = options[mode]
    return ('from CompactionAnalyzer import (\n    analyze_image,\n)\n\n'
            'value = analyze_image(\n    "orientation.tif",\n'
            f'    mode="{mode}",\n{arguments}'
            f'    outputs="{output}",\n'
            '    sigma_tensor=6,\n    edge=10,\n)')


def python_api_help(mode):
    text = ('<h2>Python API</h2>'
            '<p>Call the analysis from another Python program with '
            '<b>CompactionAnalyzer.analyze_image</b>. This example returns '
            'one number and writes no output files.</p>'
            '<pre style="font-family: Consolas, monospace; font-size: 9pt;">'
            + escape(analysis_api_example(mode)) + '</pre>'
            '<p>Use image paths or NumPy arrays. Adjust <b>sigma_tensor</b> '
            'and <b>edge</b> for your images; both are in pixels.</p>')
    if mode in ("point_orientation", "line_orientation"):
        text += ('<p>Coordinates are zero-based pixels of the original image: '
                 '<b>(0, 0)</b> is top left, X increases rightwards and Y downwards. '
                 'Fractional coordinates are accepted.</p>'
                 '<p>Optionally add <b>mask_image="mask.tif"</b> or '
                 '<b>mask=selected_area</b>. The mask selects pixels; '
                 'the point or line still defines the reference. In the GUI, choose '
                 '<b>Mask from the same image</b> or <b>Mask from a second image</b> when importing.</p>')
    if mode == "line_orientation":
        text += ('<p>Both points must be inside the image. The line defines a direction '
                 'across the whole image; reversing its endpoints gives the same result.</p>'
                 '<p>Use <b>mean_deviation_line</b> for the mean absolute angle in degrees, '
                 '<b>angle_line</b> for a signed angle map, or <b>alignment_line</b> '
                 'for the alignment map.</p>')
    elif mode == "point_orientation":
        text += ('<p>Use <b>mean_angle_point</b> for the mean absolute angle in degrees '
                 'or <b>alignment_point</b> for the radial alignment map.</p>'
                 '<p>For concentric distance shells, request <b>outputs="distance_profile"</b> '
                 'and set <b>shell_width=10</b> (pixels). The returned dictionary contains '
                 '<b>distance_px</b>, <b>mean_alignment_point</b>, <b>mean_angle_point</b> '
                 'and <b>mean_coherency</b> arrays. The mask selects pixels within each ring. '
                 'Empty rings contain NaN means and zero pixel counts. '
                 '<b>distance_um</b> uses the supplied <b>scale</b> in µm/pixel.</p>')
    elif mode == "axis_orientation_masked":
        text += ('<p>This call masks the orientation image itself. Add '
                 '<b>mask_image="mask.tif"</b> to use a second image, or '
                 '<b>mask=selected_area</b> for an existing boolean mask.</p>')
    text += ('<p>A single output name returns its number or map. A list such as '
             '<b>["mean_alignment_x", "mean_coherency"]</b> returns a dictionary. '
             'Returned maps are cropped by <b>edge</b>.</p>')
    return text
