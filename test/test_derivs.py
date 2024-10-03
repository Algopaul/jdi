import jax
import jax.numpy as jnp
from absl.testing import absltest
from functools import partial
from jdi.derivs import periodic_finite_difference, get_deriv_dict


class DerivativeTests(absltest.TestCase):

  def setUp(self):
    # Set up sample data and stepsizes
    self.x = jnp.linspace(0, 2 * jnp.pi, 300, endpoint=False)  # Domain points
    self.u = jnp.sin(self.x)
    self.stepsize = self.x[1] - self.x[0]  # Uniform stepsize

  def test_first_derivative(self):
    """Test the first-order derivative using periodic boundary conditions."""

    # Expected result for first derivative: cos(x)
    expected = jnp.cos(self.x)

    # JIT-compiled function for first derivative
    jit_first_deriv = jax.jit(
        partial(
            periodic_finite_difference,
            stepsize=float(self.stepsize),
            axis=0,
            order=1,
        ))

    # Compute first derivative
    computed = jit_first_deriv(self.u)

    # Assert that the error between computed and expected is small
    error = jnp.linalg.norm(computed - expected)
    self.assertLess(
        error,
        1e-3,
        "First derivative with JIT should approximate cos(x)",
    )

  def test_second_derivative(self):
    """Test the second-order derivative using periodic boundary conditions."""

    # Expected result for second derivative: -sin(x)
    expected = -jnp.sin(self.x)

    # JIT-compiled function for second derivative
    jit_second_deriv = jax.jit(
        partial(
            periodic_finite_difference,
            stepsize=float(self.stepsize),
            axis=0,
            order=2,
        ))

    # Compute second derivative
    computed = jit_second_deriv(self.u)

    # Assert that the error between computed and expected is small
    error = jnp.linalg.norm(computed - expected)
    self.assertLess(error, 1e-3,
                    "Second derivative with JIT should approximate -sin(x)")

  def test_derivative_dict_jit(self):
    """Test the derivative dictionary returned by get_deriv_dict with JIT compilation."""
    stepsizes = [self.stepsize, self.stepsize]
    deriv_dict = get_deriv_dict(stepsizes)

    # JIT-compiled functions from the deriv_dict
    jit_d_dx0 = jax.jit(deriv_dict['periodic_d_dx0'])
    jit_d2_dx0 = jax.jit(deriv_dict['periodic_d2_dx0'])

    # Test first derivative using the dictionary (d/dx0 ~ cos(x))
    expected_d_dx0 = jnp.cos(self.x)
    computed_d_dx0 = jit_d_dx0(self.u)
    error_d_dx0 = jnp.linalg.norm(computed_d_dx0 - expected_d_dx0)
    self.assertLess(
        error_d_dx0, 1e-3,
        "First derivative (d/dx0) from deriv_dict should approximate cos(x)")

    # Test second derivative using the dictionary (d²/dx0² ~ -sin(x))
    expected_d2_dx0 = -jnp.sin(self.x)
    computed_d2_dx0 = jit_d2_dx0(self.u)
    error_d2_dx0 = jnp.linalg.norm(computed_d2_dx0 - expected_d2_dx0)
    self.assertLess(
        error_d2_dx0, 1e-3,
        "Second derivative (d²/dx0²) from deriv_dict should approximate -sin(x)"
    )


if __name__ == '__main__':
  absltest.main()
