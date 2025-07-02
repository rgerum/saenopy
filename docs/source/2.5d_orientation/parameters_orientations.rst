Orientation
===================

sigma_tensor
----------------
Windowsize, in which individual structure elements are calculated. 
Should be in the range of the underlying structure and can be optimised per setup by performing a test-series. 


edge
----------------
Pixel width of the edges left blank because alignment cannot be calculated there. Default is 20 px.


max_dist
-----------
(Optional) Specify a maximum distance around the cell center for analysis (in px). Default is None.


ignore_cell_outline
-----------
By default, the cell-occupied area is excluded, and matrix alignment is calculated only in the remaining area.
To analyze alignment across the entire field of view (e.g., relative to the x- or y-axis), set this parameter to True. Default is False.


sigma_first_blur
-----------
Initial slight Gaussian blur applied to the fiber image before structure analysis. Default is 0.5.

angle_sections
-----------
Angle sections around the cell in degrees, used for polarity analysis of matrix fiber orientation. Default is 5°.

shell_width
-----------
Distance shells around the cell for analyzing intensity and orientation propagation over distance. Default is 5 µm.