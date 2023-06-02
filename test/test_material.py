from saenopy.materials import Material, LinearMaterial, SemiAffineFiberMaterial
import numpy as np
from findiff import FinDiff
import pytest


def test_material():
    material = Material()
    with pytest.raises(NotImplementedError):
        material.stiffness([])
    with pytest.raises(NotImplementedError):
        material.energy([])
    with pytest.raises(NotImplementedError):
        material.force([])

    material_list = [
        SemiAffineFiberMaterial(900, 0.0004, 0.0075, 0.033),
        SemiAffineFiberMaterial(900, None, 0.0075, 0.033),
        SemiAffineFiberMaterial(900, None, None, 0.033),
        SemiAffineFiberMaterial(900, 1e-31, 0.0075, None),
        LinearMaterial(900),
    ]
    derivative = FinDiff(0, 0.0001)

    gamma = np.arange(0.005, 0.3, 0.0001)
    gamma2 = np.arange(-0.5, 0.3, 0.0001)
    for material in material_list:
        print(material)
        lookup = material.generate_look_up_table()
        lookup.py_func(gamma2)

        s = material.stiffness(gamma)
        f = material.force(gamma)

        e_prime = derivative(material.energy(gamma))
        f_prime = derivative(material.force(gamma))

        np.testing.assert_almost_equal(np.log(f), np.log(e_prime), decimal=3)
        np.testing.assert_almost_equal(np.log(s), np.log(f_prime), decimal=3)

