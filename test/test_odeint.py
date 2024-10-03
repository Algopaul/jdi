import jax.numpy as jnp
from absl.testing import absltest
from jdi.odeint import OdeSolver


# Define a simple harmonic oscillator ODE
def harmonic_oscillator_rhs(t, x):
  return jnp.array([x[1], -x[0]])


class OdeSolverTest(absltest.TestCase):

  def test_rk4_convergence(self):
    """Tests if RK4 method achieves expected convergence rate for a known ODE."""

    # True solution: sine and cosine functions
    def true_solution(t):
      return jnp.array([jnp.cos(t), -jnp.sin(t)])

    # Time settings
    timepoints = jnp.linspace(0, jnp.pi, 20)
    x0 = jnp.array([1.0, 0.0])  # Initial condition: [cos(0), -sin(0)]

    # Create the solver and test convergence by comparing against exact solution
    solver = OdeSolver(harmonic_oscillator_rhs)
    result = solver.solve(x0, timepoints, supersample_factor=1)
    exact_solution = true_solution(timepoints)

    # Compute the error between the numerical and exact solutions
    error = jnp.linalg.norm(result[-1] - exact_solution.T[-1], axis=0)
    self.assertLess(float(jnp.max(error)), 1e-4)

  def test_subsampling(self):
    """Tests if subsampling correctly computes every n-th point."""
    supersample_factor = 10

    # Time settings
    timepoints = jnp.linspace(0, jnp.pi, 20)
    x0 = jnp.array([1.0, 0.0])

    # Create solver with subsampling
    solver = OdeSolver(harmonic_oscillator_rhs)
    result_subsampled = solver.solve(
        x0,
        timepoints,
        supersample_factor=supersample_factor,
    )
    result_pre = solver.solve(
        x0,
        timepoints,
        supersample_factor=1,
    )
    error = jnp.linalg.norm(result_pre[-1] - result_subsampled[-1], axis=0)
    self.assertLess(float(jnp.max(error)), 1e-4)

    self.assertEqual(
        result_subsampled.shape[0],
        len(timepoints),
        "Subsampled output has incorrect length",
    )


if __name__ == '__main__':
  absltest.main()
