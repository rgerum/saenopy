.. _SectionMesh:

Mesh
====

Mesh definition
---------------

Saenopy uses only thetrahedral meshes. The mesh is defined by the N nodes
and by the connectivity of the nodes by M thetrahedra.

Text files
~~~~~~~~~~

.. container:: twocol

    .. container:: leftside

        Nodes are an Nx3 float array (three spatial dimensions). Units are in meters.

        Nodes can be loaded from a .txt file structured like this

        .. literalinclude:: nodes.txt
           :caption:
           :linenos:

    .. container:: rightside

        Connectivity is an Mx4 integer array (reference the node indices of the 4 corners).

        The connectivity can be loaded from a .txt file structured like this

        .. literalinclude:: connectivity.txt
           :caption:
           :linenos:

.. raw:: html

    <div style="clear: both;"></div>

Both can be loaded using `np.loadtxt` and added to the solver using :py:meth:`~.Solver.setNodes` and
:py:meth:`~.Solver.setTetrahedra`.

.. code-block:: python
    :linenos:

    import numpy as np
    from saenopy import Solver

    # initialize the solver
    M = Solver()
    # load the nodes (units in meters)
    M.setNodes(np.loadtxt("nodes.txt"))
    # load the connectivity
    M.setTetrahedra(np.loadtxt("connectivity.txt"))

Gmsh file
~~~~~~~~~
If the mesh was created in gmsh, saenopy provides a loader to directly load files of the `Gmsh format <http://gmsh.info/>`_.

.. code-block:: python
    :linenos:

    from saenopy import load

    # load gmsh file and return a solver object with the mesh
    M = load.load_gmsh("mesh.msh")


Defining Inputs
---------------

.. _SectionBoundaryConditions:

Boundary Conditions
~~~~~~~~~~~~~~~~~~~

For using the :ref:`SectionBoundaryConditionMode`, constraints for displacement (Nx3 array) and force (Nx3 array) have to be provided.
For each node, either a displacement or a force needs to be given, the other has to be nan.

- For a fixed node
    - the displacement should be provided (3 float values, in meters)
    - the force should left open (3 nan values)
- For a free node,
    - the displacement should be left open (3 nan values)
    - the force should provided (3 float values, in Newtons).


.. container:: twocol

    .. container:: leftside

        .. literalinclude:: constraint_displacement.txt
           :caption:
           :linenos:

    .. container:: rightside

        .. literalinclude:: constraint_force.txt
           :caption:
           :linenos:

.. raw:: html

    <div style="clear: both;"></div>


Both can be loaded using `np.loadtxt` and added to the solver using :py:meth:`~.Solver.setBoundaryCondition`.

.. code-block:: python
    :linenos:

    # load the displacement constraints (in meters)
    node_displacement = np.loadtxt("constraint_displacement.txt")
    # load the force constraints (in Newton)
    node_force = np.loadtxt("constraint_force.txt")
    # hand the boundary conditions to the solver
    M.setBoundaryCondition(node_displacement, node_force)

.. _SectionMeasuredDisplacement:

Measured displacement
~~~~~~~~~~~~~~~~~~~~~

For using :ref:`SectionRegularizationMode`, the measured (or target) displacement (Nx3 array, in meters) has to be provided for
all nodes.

.. literalinclude:: measured_displacement.txt
   :caption:
   :linenos:

It can be loaded using `np.loadtxt` and added to the solver using :py:meth:`~.Solver.setTargetDisplacements`.

.. code-block:: python
    :linenos:

    node_displacement = np.loadtxt("measured_displacement.txt")
    M.setTargetDisplacements(node_displacement)


.. raw:: html

    <div id="root"></div>

    <script type="importmap">
      {
        "imports": {
          "three": "https://unpkg.com/three@v0.158.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@v0.158.0/examples/jsm/"
        }
      }
    </script>
    <script type="module">
      import { init } from "./_static/js/3d_viewer.mjs";
      init({
        path: "./_static/vector_data2",
        scale: 1,
        dom_node: document.getElementById("root"),
        zoom: 1.5,
        image: "z-pos",
        mouse_control: false,
        animations: [{type: "scroll-tilt"}]
      });
    </script>
