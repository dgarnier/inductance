"""See if we can derive an expression for the gmd of a triangle."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# from functools import partial


def normalized_triangle_pts(a, b):
    """Return a triangle with normalized sides a and b.

    The triangle is normalized so that the half perimeter is 1.
    This means all shapes are represented by the two sides a and b
    which can have a range of values from 0 to 1.  This will
    capture all possible triangles that have the same shape. Of course,
    one must allow for change is scale, location, and orientation, and
    reflection.

    Will return points in a counter-clockwise order only. Starting at origin.
    and the side a along the x-axis.

    This equation can be found by setting a + b + c = 2 and using the law of
    cosines to solve for the cosine of the angle opposite side c, gamma.

                                    + (a-mx-b*cos(g), b*sin(g)-my)
                                  .  .
                                .     .
                           c  .        .  b
                            .           .
                          .            g .
             (-mx, -my) + - - - - - - - - + (a-mx, -my)
                                a

    Args:
        a (float): length of side a
        b (float): length of side b

    Returns:
        tuple(tuple(float)): points of triangle in counter-clockwise order
    """
    # pts = np.zeros((3,2), dtype=np.float64)
    cos_g = (2 * (a + b) - (2 + a * b)) / (a * b)
    sin_g = jnp.sqrt(1 - cos_g**2)

    mx = (2 * a - b * cos_g) / 3
    my = b * sin_g / 3

    pts = np.array(
        [
            [-mx, -my],
            [a - mx, -my],
            [a - mx - b * cos_g, b * sin_g - my],
        ],
        dtype=np.float64,
    )
    return pts


def normalize_triangle(pts):
    """Convert from points to normalized sides."""
    a = np.sqrt((pts[0, 0] - pts[1, 0]) ** 2 + (pts[0, 1] - pts[1, 1]) ** 2)
    b = np.sqrt((pts[1, 0] - pts[2, 0]) ** 2 + (pts[1, 1] - pts[2, 1]) ** 2)
    c = np.sqrt((pts[2, 0] - pts[0, 0]) ** 2 + (pts[2, 1] - pts[0, 1]) ** 2)

    s = (a + b + c) / 2
    return jnp.array([a, b], dtype=jnp.float64) / s


def point_in_triangle(pt, tri):
    """Return True if pt is in tri, False otherwise."""
    inside = pt[1] < tri[0, 1]
    inside &= (
        (pt[1] - tri[1, 1]) * (tri[2, 0] - tri[1, 0])
        - (pt[0] - tri[1, 0]) * (tri[2, 1] - tri[1, 1])
    ) >= 0.0
    inside &= (
        (pt[1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0])
        - (pt[0] - tri[2, 0]) * (tri[0, 1] - tri[2, 1])
    ) >= 0.0
    return inside


@jax.jit
def points_in_triangle(pts, tri):
    """Return True if pts are in tri, False otherwise."""
    result = jnp.zeros(pts.shape[0], dtype=jnp.bool_)
    for i in range(pts.shape[0]):
        result = result.at[i].set(point_in_triangle(pts[i, :], tri))
    return result


def random_pts_in_tri(tri, bounds, key):
    """Return two random points in the triangle defined by tri."""
    [[x_min, x_max], [y_min, y_max]] = bounds
    while True:
        key, *sks = jax.random.split(key, 3)
        pt1 = random.uniform(sks, (2,), minval=bounds[0, :], maxval=bounds[1, :])
        if point_in_triangle(pt1, tri):
            break
    while True:
        key, *sks = jax.random.split(key, 3)
        pt2 = random.uniform(sks, (2,), minval=bounds[0, :], maxval=bounds[1, :])
        if point_in_triangle(pt2, tri):
            break
    return jnp.array([pt1, pt2], dtype=np.float64)


@jax.jit
def integrate_mean_distance(a, b, npts):
    """Calculate the mean distance between two random points in a triangle."""
    tri = normalized_triangle_pts(a, b)
    bounds = jnp.array(
        [[tri[:, 0].min(), tri[:, 0].max()], [tri[:, 1].min(), tri[:, 1].max()]]
    )

    key = random.PRNGKey(0)
    sum = 0.0

    # TODO : replace with jax.vmap
    for _ in range(npts):
        key, subkey = random.split(key)
        pts = random_pts_in_tri(tri, bounds, subkey)
        sum += math.log((pts[0, 0] - pts[1, 0]) ** 2 + (pts[0, 1] - pts[1, 1]) ** 2)
    return math.exp(sum / npts / 2)


def create_pts_and_check(tri, npts):
    """Create npts random points in the triangle defined by tri."""
    bounds = np.array(
        [[tri[:, 0].min(), tri[:, 0].max()], [tri[:, 1].min(), tri[:, 1].max()]]
    )

    key = random.PRNGKey(0)
    key, sks = random.split(key, 2)
    pts = random.uniform(sks, (npts, 2), dtype=jnp.float64)
    pts = pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    inp = points_in_triangle(pts, tri)
    # inp = jnp.zeros(npts, dtype=jnp.bool_)
    # for i in range(npts):
    #    inp = inp.at[i].set(point_in_triangle(pts[i, :], tri))
    return pts, inp


def plot_pts_in_triangle(a, b, npts):
    """Plot the points in the triangle defined by a and b."""
    npts = jnp.int32(npts)
    tri = normalized_triangle_pts(a, b)
    pts, inp = create_pts_and_check(tri, npts)

    plt.plot(tri[(0, 1, 2, 0), 0], tri[(0, 1, 2, 0), 1], "k-")
    plt.gca().set_aspect("equal")
    plt.plot(pts[inp, 0], pts[inp, 1], "g,")
    plt.plot(pts[~inp, 0], pts[~inp, 1], "r,")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jax import config

    config.update("jax_enable_x64", True)

    # plot_pts_in_triangle(2 / 3, 2 / 3, 10000)

    # Weaver calculates the g.m.d. of an equilateral triangle with sides = 1
    # to be 0.308382
    # with 10^8 points, i get 0.3083905564333403

    print(
        "g.m.d of equilateral triangle with sides = 1: ",
        integrate_mean_distance(2 / 3, 2 / 3, npts=10000 * 100000) * 3 / 2,
    )
