Material Parameters
===================
The material parameters :math:`k`, :math:`d_0`, :math:`\lambda_s`, and :math:`d_s` are explained in the section
:ref:`SectionMaterial`.

.. figure:: images/theory/material/fiber.png
    :width: 70%

Meaning of the parameters
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Parameter
     - Meaning
   * - :math:`k`
     - Stiffness of the material in the linear regime.
   * - :math:`d_0`
     - Decay parameter in the buckling regime. If omitted, the material shows no
       buckling but a linear response under compression.
   * - :math:`\lambda_s`
     - Stretch at which strain stiffening starts. If omitted, the material shows
       no strain stiffening.
   * - :math:`d_s`
     - How strong the strain stiffening is. If omitted, the material shows no
       strain stiffening.

Setting only :math:`k` and omitting the other three gives a linear material,
which is what :py:class:`~saenopy.materials.LinearMaterial` does.

Values used in the examples
---------------------------

These are the parameter sets the bundled examples run with. They are starting
points, not reference values for collagen in general: the parameters depend
strongly on concentration, batch and polymerisation conditions, so for your own
gel they have to be fitted to a rheological measurement or taken from a
publication that characterised the same preparation.

.. list-table::
   :header-rows: 1
   :widths: 34 13 13 13 13 14

   * - Used by
     - :math:`k`
     - :math:`d_0`
     - :math:`\lambda_s`
     - :math:`d_s`
     - Notes
   * - Single cell, organoid and brightfield examples
     - 6062
     - 0.0025
     - 0.0804
     - 0.034
     - the default in most examples
   * - Dynamical single cell example (NK cells)
     - 1449
     - 0.0022
     - 0.032
     - 0.055
     - described in the interface as collagen I, 1.2 mg/ml
   * - Linear material
     - Young's modulus :math:`\times` 6
     - —
     - —
     - —
     - :py:class:`~saenopy.materials.LinearMaterial`, by definition
