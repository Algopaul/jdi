"""Utils for derivative computation"""
from functools import partial
import jax.numpy as jnp
from jax.typing import ArrayLike


def periodic_finite_difference(u: ArrayLike,
                               stepsize: float,
                               axis: int = 0,
                               order: int = 1):
  if order == 1:
    return 1 / (2 * stepsize) * (jnp.roll(u, -1, axis) - jnp.roll(u, 1, axis))
  elif order == 2:
    return 1 / (stepsize**2) * (
        jnp.roll(u, -1, axis) + jnp.roll(u, 1, axis) - 2 * u)
  elif order == 3:
    factor = 1 / (stepsize**3)
    p1 = 1 / 8 * (jnp.roll(u, 3, axis) - jnp.roll(u, -3, axis))
    p2 = 1 * (-jnp.roll(u, 2, axis) + jnp.roll(u, -2, axis))
    p3 = 13 / 8 * (jnp.roll(u, 1, axis) - jnp.roll(u, -1, axis))
    return factor * (p1 + p2 + p3)
  else:
    raise NotImplementedError('Higher orders not implemented')


def get_deriv_dict(stepsizes):
  deriv_dict = {
      'periodic_d_dx0':
          partial(
              periodic_finite_difference,
              stepsize=stepsizes[0],
              axis=0,
              order=1),
      'periodic_d_dx1':
          partial(
              periodic_finite_difference,
              stepsize=stepsizes[1],
              axis=1,
              order=1),
      'periodic_d2_dx0':
          partial(
              periodic_finite_difference,
              stepsize=stepsizes[0],
              axis=0,
              order=2),
      'periodic_d2_dx1':
          partial(
              periodic_finite_difference,
              stepsize=stepsizes[1],
              axis=1,
              order=2)
  }
  return deriv_dict
