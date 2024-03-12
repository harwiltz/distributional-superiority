import unittest

import jax
import jax.numpy as jnp

from dsup import statistical_functionals


class TestRiskMeasure(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(42)
        self.num_atoms = 1000

    def test_cvar_1_mean_consistency(self):
        data = jax.random.normal(self.rng, (self.num_atoms,))
        mean = jnp.mean(data)
        risk_measure = statistical_functionals.CVaRFunctional(1.0)
        cvar = risk_measure.evaluate(data)
        self.assertAlmostEqual(cvar, mean)

    def test_cvar_coarse(self):
        n_zeros = 30
        n_ones = 70
        data = jnp.concatenate([jnp.zeros(n_zeros), jnp.ones(n_ones)])
        cvar_low = statistical_functionals.CVaRFunctional(
            (n_zeros - 1) / (n_zeros + n_ones)
        )
        self.assertEqual(cvar_low.evaluate(data), 0.0)

    def test_cvar_coarse_sorting(self):
        n_zeros = 30
        n_ones = 70
        data_ood = jnp.concatenate([jnp.ones(n_ones), jnp.zeros(n_zeros)])
        alpha = (n_zeros - 1) / (n_zeros + n_ones)
        cvar_nosort = statistical_functionals.CVaRFunctional(alpha)
        self.assertEqual(cvar_nosort.evaluate(data_ood), 1.0)
        cvar_sort = statistical_functionals.CVaRFunctional(alpha, requires_sort=True)
        self.assertEqual(cvar_sort.evaluate(data_ood), 0.0)


if __name__ == "__main__":
    unittest.main()
