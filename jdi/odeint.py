"""Tools for integrating ODEs on a given regular time grid"""
import jax
import jax.numpy as jnp
from absl import logging


class OdeSolver:
  """ODE solver class that supports different integration methods and subsampling."""

  def __init__(self, ode_rhs, bc_callback=lambda x: x):
    self.ode_rhs = ode_rhs
    self.bc_callback = bc_callback

  def solve(
      self,
      initial_condition,
      timepoints,
      method="rk4",
      supersample_factor=1,
  ):
    logging.info('Starting ODE solver with method: %s, supersample_factor: %d',
                 method, supersample_factor)
    if supersample_factor > 1:
      return self._solve_subsampled(initial_condition, timepoints,
                                    supersample_factor, method)
    return self._solve_full(initial_condition, timepoints, method)

  def _solve_full(self, initial_condition, timepoints, method):
    """Full integration without subsampling."""
    logging.info('Performing full integration without subsampling')
    step_fn = self._get_step_function(method)
    dt = timepoints[1] - timepoints[0]

    def inner_step(carry, t):
      x = step_fn(carry, t, t + dt)
      x = self.bc_callback(x)
      return x, x

    _, time_series = jax.lax.scan(inner_step, initial_condition,
                                  timepoints[:-1])
    return jnp.stack((initial_condition, *time_series))

  def _solve_subsampled(self, initial_condition, timepoints, supersample_factor,
                        method):
    logging.info('Performing integration with subsampling factor %d',
                 supersample_factor)
    step_fn = self._get_step_function(method)
    t = self._supersample_timepoints(timepoints, supersample_factor)

    dt = t[1] - t[0]

    def fori_step(i, carry):
      x, t = carry
      xp = step_fn(x, t, t + dt)
      xp = self.bc_callback(xp)
      return (xp, t + dt)

    def scan_step(xp, t):
      xp, _ = jax.lax.fori_loop(0, supersample_factor, fori_step, (xp, t))
      return xp, xp

    t_sub = t[:-1:supersample_factor]
    _, time_series = jax.lax.scan(scan_step, initial_condition, t_sub)
    return jnp.stack((initial_condition, *time_series))

  def _get_step_function(self, method):
    if method == "rk4":
      return self._rk4_step
    else:
      raise ValueError(f"Unknown integration method: {method}")

  def _rk4_step(self, x, t0, t1):
    """Fourth-order Runge-Kutta integration step."""
    delta_t = t1 - t0
    k1 = self.ode_rhs(t0, x)
    k2 = self.ode_rhs(t0 + delta_t / 2, x + delta_t * k1 / 2)
    k3 = self.ode_rhs(t0 + delta_t / 2, x + delta_t * k2 / 2)
    k4 = self.ode_rhs(t1, x + delta_t * k3)
    return x + 1.0 / 6.0 * delta_t * (k1 + 2 * k2 + 2 * k3 + k4)

  def _supersample_timepoints(self, timepoints, supersample_factor):
    """Generate supersampled timepoints."""
    num_samples = len(timepoints) + (len(timepoints) - 1) * (
        supersample_factor - 1)
    return jnp.linspace(timepoints[0], timepoints[-1], num_samples)
