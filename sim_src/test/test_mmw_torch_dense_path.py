import numpy as np

from sim_src.alg.mmw import mmw


def test_build_dense_x_blocks_matches_numpy_reference():
    alg = mmw(nit=1, device="cpu")

    x_half = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    gain_x = np.array([0, 1], dtype=int)
    gain_y = np.array([1, 2], dtype=int)
    asso_x = np.array([0], dtype=int)
    asso_y = np.array([2], dtype=int)

    x_mdiag_data, x_offdi_s, x_offdi_q = alg._build_dense_x_blocks(
        x_half,
        gain_x,
        gain_y,
        asso_x,
        asso_y,
    )

    np.testing.assert_allclose(x_mdiag_data, np.array([1.0, 0.5, 1.0]))
    np.testing.assert_allclose(x_offdi_s, np.array([0.5, 0.5]))
    np.testing.assert_allclose(x_offdi_q, np.array([0.0]))

