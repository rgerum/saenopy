Command Line Interface
======================

Saeno can be used as a python package but can also be used as a command line tool:

.. code-block:: ini

   saeno CONFIG config.txt

The `config.txt` file can contain all the parameters that the execution needs.

Relaxation Mode
---------------

.. code-block:: ini
   :linenos:

   # the mode
   MODE = relaxation
   BOXMESH = 0
   FIBERPATTERNMATCHING = 0

   # the solver parameters
   REL_CONV_CRIT = 1e-11
   REL_ITERATIONS = 300
   REL_SOLVER_STEP = 0.066

   # the material model
   K_0 = 1645
   D_0 = 0.0008
   L_S = 0.0075
   D_S = 0.033

   # the inputs
   COORDS = coords.dat
   TETS = tets.dat
   BCOND = bcond.dat
   ICONF = iconf.dat

   # where to save the output
   DATAOUT = output

Regularization Mode
-------------------

.. code-block:: ini
   :linenos:

   # the mode
   MODE = regularization
   BOXMESH = 0
   FIBERPATTERNMATCHING = 0

   # the solver parameters
   REG_CONV_CRIT = 0.01
   REG_ITERATIONS = 50
   REG_RELREC = relrec.dat
   REG_SIGMAZ = 1.0
   REG_SOLVER_PRECISION = 1e-18
   REG_SOLVER_STEP = 0.33

   # the material model
   K_0 = 1645
   D_0 = 0.0008
   L_S = 0.0075
   D_S = 0.033

   # the inputs
   COORDS = coords.dat
   TETS = tets.dat
   UFOUND = Ufound.dat
   SFOUND = Sfound.dat

   # where to save the output
   DATAOUT = output
