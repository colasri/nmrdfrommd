from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader

TWOMOL_DIR = Path(__file__).parent / "datafiles" / "twomolecules"


@pytest.fixture
def universe_small():
    coords = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0]],
        ]
    )

    u = mda.Universe.empty(3, trajectory=True)
    u.add_TopologyAttr("name", ["H", "H", "H"])

    u.load_new(coords, format=MemoryReader)

    ag = u.atoms

    return u, ag


@pytest.fixture
def twomol_files():
    """Paths to the two-molecule LAMMPS data file and XTC trajectory."""
    return TWOMOL_DIR / "twomolecules.data", TWOMOL_DIR / "twomolecules.xtc"


@pytest.fixture
def twomol_universe(twomol_files):
    u = mda.Universe(*twomol_files)
    group_H2O = u.select_atoms("type 2")  # pick one H per molecule
    return u, group_H2O


@pytest.fixture
def two_molecule_system():
    """
    3 atoms in residue 1 (atoms 0-2), 3 atoms in residue 2 (atoms 3-5)
    Molecule 1: H H O
    Molecule 2: H H O
    """
    resids = np.array([1, 1, 1, 2, 2, 2])
    indices = np.array([0, 1, 2, 3, 4, 5])
    atom_types = np.array(["H", "H", "O", "H", "H", "O"])
    return resids, indices, atom_types


# Orthorhombic box: lengths in Angstrom, angles in degrees.
DEFAULT_BOX = (50.0, 50.0, 50.0, 90.0, 90.0, 90.0)
RIGID_DIPOLE_DISTANCE = 2.0  # Angstrom


def _build_universe(coords, names, resids, dt=1.0, box=DEFAULT_BOX):
    """Build an in-memory universe.

    Parameters
    ----------
    coords : array-like, shape (n_frames, n_atoms, 3)
        Positions in Angstrom.
    names : list of str
        Atom names, also used as elements.
    resids : list of int
        Residue (molecule) id of each atom.
    dt : float
        Time between frames, in ps.
    box : sequence of 6 floats or None
        Box dimensions; None gives a trajectory without box information.
    """
    coords = np.asarray(coords, dtype=np.float32)
    residues = sorted(set(resids))
    u = mda.Universe.empty(
        coords.shape[1],
        n_residues=len(residues),
        atom_resindex=[residues.index(r) for r in resids],
        trajectory=True,
    )
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("elements", names)
    u.add_TopologyAttr("resid", residues)
    u.load_new(coords, format=MemoryReader, dt=dt, dimensions=box)
    return u


@pytest.fixture
def make_universe():
    """Factory for an in-memory universe; see _build_universe for the arguments."""
    return _build_universe


@pytest.fixture
def make_rigid_pair(make_universe):
    """Factory for a universe of two atoms frozen RIGID_DIPOLE_DISTANCE apart.

    The pair is aligned along z (theta = 0) and never moves. Call it as
    ``make_rigid_pair(names=("H", "F"), dt=..., box=...)``.
    """

    def make(names=("H", "H"), dt=1.0, box=DEFAULT_BOX, n_frames=64):
        coords = np.zeros((n_frames, 2, 3))
        coords[:, :] = [25.0, 25.0, 25.0]
        coords[:, 1, 2] += RIGID_DIPOLE_DISTANCE
        return make_universe(coords, list(names), [1, 1], dt=dt, box=box)

    return make


@pytest.fixture
def rigid_dipole_universe(make_rigid_pair):
    """Two H atoms at a fixed distance along z, frozen over the whole trajectory.

    The dipole never moves, so the m=0 correlation function is a constant
    equal to (3 cos^2(theta) - 1)^2 / r^6 = 4 / r^6 (theta = 0).
    """
    u = make_rigid_pair()
    return u, u.atoms, RIGID_DIPOLE_DISTANCE
