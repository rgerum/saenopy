Segmentation
=========================

thres
-----

An additional multiplier applied to the Otsu threshold for determining the cell outline from the cell images. Default is 1. 
Use the invert option to segment a dark cell (e.g., in bright-field images) instead of a bright one (e.g., in fluorescence images).

gauss1,2
-----
A pair of Gaussian filters applied before segmentation, forming a bandpass filter. Defaults are 0.5 and 100.
