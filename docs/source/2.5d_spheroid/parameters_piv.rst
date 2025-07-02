Detect Deformations
===================

window size
----------------

This is the size of the window used to detect matrix deformations using Particle Image Velocimetry (PIV). This size is given in pixels. 
Deformations are detected between consecutive timesteps and accumulated over time per window.

overlap
----------------

This defines how adjacent windows overlap, and adjusts the number of visible arrows in the deformation field. The default value is 50%.

n_min, n_max
-----------

These are the minimum and maximum image indices taken into account for the analysis. These can be used to skip the initial and latest images in a time series.

segmentation threshold
---------------
This threshold is multiplied by the Otsu threshold to fine-tune the detection of the spheroid outline.


continous_segmentation
----------------
If set to true, forces are calculated by considering the spheroid area at each timestep.
By default, this is set to False, meaning that the forces are calculated based on the initial spheroid size over time. 
This avoids possible fluctuations caused by jumps in segmentation.