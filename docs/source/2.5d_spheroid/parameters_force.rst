Calculate Forces
=========================

Lookup Table
-----

These are material-specific lookup tables that show, for a given force, how the matrix deformation around a spheroid decays over distance for 
a particular material. These tables are calculated using finite-element simulations of a spherical inclusion under different forces and allow to assign 
the best-fitting force for each deformationfield by the course of matrix deformations over distance. Tables can be generated for linear or non-linear 
elastic materials. Existing tables for collagen and linear elastic materials can be found here:

https://github.com/christophmark/jointforces/tree/master/docs/data

More information about the look-up tables can be found here:

https://elifesciences.org/articles/51912


r_min, r_max
-----
These parameters can be used to specify a range from r_min to r_max (or 'None' for no restriction), defining which deformations should be included in the force reconstruction. This is useful, for example, 
when the shape is not perfectly spherical and near-matrix deformations should be excluded to focus on the far-field region. Units are in effective spheroid radii calculated from the spheroid segmentation.





