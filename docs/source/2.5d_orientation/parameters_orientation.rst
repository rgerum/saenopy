Orientation
===================

sigma_tensor
----------------
Windowsize, in which individual structure elements are calculated. 
Should be in the range of the underlying structure and can be optimised per setup by performing a test-series. 


edge
----------------
Pixel width of the edges left blank because alignment cannot be calculated there. The GUI default is 40 px. In axis mode, zero keeps the whole image.


max_dist
-----------
(Optional) Specify a maximum distance around the cell center for analysis (in px). Default is None.


ignore_cell_outline
-------------------
By default, the cell-occupied area is excluded, and matrix alignment is calculated only in the remaining area.
This option belongs to cell-relative analysis and retains its cell-center reference. For pure axis orientation, choose the axis workflow and optionally use the whole image instead. Default is False.


sigma_first_blur
----------------
Initial slight Gaussian blur applied to the fiber image before structure analysis. Default is 0.5.

angle_sections
--------------
Angle sections around the cell in degrees, used for polarity analysis of matrix fiber orientation. Default is 5°.

shell_width
-----------
Distance shells around the cell for analyzing intensity and orientation propagation over distance. Default is 5 µm.

Axis validity mask
------------------
These controls are shown only for masked axis measurements. They are independent
of the full workflow's cell-segmentation parameters.

* ``mask_sigma``: Gaussian blur of the raw, edge-cropped second channel, in
  pixels (default 5).
* ``mask_threshold``: factor inside ``threshold_otsu(filtered * factor)``
  (default 0.6). Select pixels with ``filtered > threshold``. This retains the
  original actinin-reader calculation, including histogram discretization.
* ``mask_invert``: select the complement (dark regions), default False.

All selected regions are retained. There is no dilation, hole filling or
largest-cell selection. Undefined fiber directions are excluded even inside
the mask. An empty selection gives NaN alignment means and zero histogram
counts. Alignment is the mean of ``1 - abs(angle_degrees)/45`` over valid pixels,
not the arithmetic mean of signed angles and not a cosine-based order parameter.
