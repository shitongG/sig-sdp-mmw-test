import numpy as np

from sim_src.env.env import env


def test_pair_conflict_matrix_is_square_and_diagonal_free():
    e = env(
        cell_size=1,
        pair_density_per_m2=0.05,
        seed=3,
        radio_prob=(1.0, 0.0),
    )

    conflict = e.build_pair_conflict_matrix()

    assert conflict.shape == (e.n_pair, e.n_pair)
    assert conflict.dtype == bool
    assert np.all(np.diag(conflict) == 0)
