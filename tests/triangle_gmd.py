"""See if we can derive an expression for the gmd of a triangle."""

# import math

import numpy as np

from inductance import _numba as nb


@nb.njit
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
                          .  bet     gam .
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
    sin_g = np.sqrt(1 - cos_g**2)

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


def scale_to_triangle(a, b, XY):
    """Generate scaling factor to make affine tranformation from cube to triangle."""
    # c**2 = a**2 + b**2 - 2*a*b*cos(gam)
    # b**2 = a**2 + c**2 - 2*a*c*cos(bet)
    # a + b + c = 2  -> c = 2 - (a + b)
    # by inspection
    # c * cos_bet + b * cos_gam = a
    # cos_bet = (a - b * cos_gam) / c

    b_cos_gam = (2 * (a + b) - (2 + a * b)) / a
    b_sin_gam = np.sqrt(b**2 - b_cos_gam**2)

    X = XY[:, 0]
    Y = XY[:, 1]

    dx = a - Y * b_cos_gam
    dy = XY[:, 0] * b_sin_gam

    x = X * dx
    y = Y * dy

    return x, y, dx, dy


def integrate_triangle_area_scaled(a, b, npts):
    """Provide a check on the triangle area."""
    pts = np.random.uniform(0, 1, size=(npts, 2))
    x, y, dx, dy = scale_to_triangle(a, b, pts)
    area_calc = np.sum(dx * dy)
    a_tri = np.sqrt((1 - a) * (1 - b) * (a + b - 1))  # heron's formula
    return area_calc / a_tri


@nb.njit(parallel=True, cache=True)
def log_gmd_integral_scaled(a, b, npts):
    """Integrate over random points in the triangle.

    This works by mapping points in a square to point in a triangle.
    Then the area is also dealt with.
    """
    a_tri = np.sqrt((1 - a) * (1 - b) * (a + b - 1))  # heron's formula
    b_cos_gam = (2 * (a + b) - (2 + a * b)) / a
    b_sin_gam = np.sqrt(b**2 - b_cos_gam**2)

    sum = 0.0
    for _i in nb.prange(npts):
        XY = np.random.uniform(0, 1, size=(2, 2))
        dx = a - XY[:, 1] * b_cos_gam
        dy = XY[:, 0] * b_sin_gam
        x = XY[:, 0] * dx
        y = XY[:, 1] * dy
        sum += (
            np.log((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
            * dx[0]
            * dy[0]
            * dx[1]
            * dy[1]
        )

    return sum / npts / 2 / a_tri**2


def normalize_triangle(pts):
    """Convert from points to normalized sides."""
    a = np.sqrt((pts[0, 0] - pts[1, 0]) ** 2 + (pts[0, 1] - pts[1, 1]) ** 2)
    b = np.sqrt((pts[1, 0] - pts[2, 0]) ** 2 + (pts[1, 1] - pts[2, 1]) ** 2)
    c = np.sqrt((pts[2, 0] - pts[0, 0]) ** 2 + (pts[2, 1] - pts[0, 1]) ** 2)

    s = (a + b + c) / 2
    return np.array([a, b]) / s


@nb.njit
def point_in_triangle(pt, tri):
    """Return True if pt is in tri, False otherwise."""
    if pt[1] < tri[0, 1]:
        return False
    left = (pt[1] - tri[1, 1]) * (tri[2, 0] - tri[1, 0]) - (pt[0] - tri[1, 0]) * (
        tri[2, 1] - tri[1, 1]
    )
    if left < 0.0:
        return False
    left = (pt[1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0]) - (pt[0] - tri[2, 0]) * (
        tri[0, 1] - tri[2, 1]
    )
    return left >= 0.0


@nb.guvectorize(["(float64[:,:], float64[:,:], bool_[:])"], "(n,m),(i,j)->(n)")
def points_in_triangle(pts, tri, res):
    """Return True if pt is in tri, False otherwise.

    Assumes tri is a triangle in counter-clockwise order.
    Assumes the first side is along the x-axis.
    """
    for i in range(pts.shape[0]):
        if pts[i, 1] < tri[0, 1]:
            res[i] = False
            continue
        left = (pts[i, 1] - tri[1, 1]) * (tri[2, 0] - tri[1, 0]) - (
            pts[i, 0] - tri[1, 0]
        ) * (tri[2, 1] - tri[1, 1])
        if left < 0.0:
            res[i] = False
            continue
        left = (pts[i, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0]) - (
            pts[i, 0] - tri[2, 0]
        ) * (tri[0, 1] - tri[2, 1])
        res[i] = left >= 0.0


@nb.njit
def random_pts_in_tri(tri, bounds=None):
    """Generate 2 random points in a triangle."""
    if bounds is None:
        x_min, x_max = tri[:, 0].min(), tri[:, 0].max()
        y_min, y_max = tri[:, 1].min(), tri[:, 1].max()
    else:
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]

    while True:
        pt1 = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
        if point_in_triangle(pt1, tri):
            break
    while True:
        pt2 = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
        if point_in_triangle(pt2, tri):
            break
    return np.array([pt1, pt2], dtype=np.float64)


@nb.njit(parallel=True, cache=True)
def log_gmd_integral(a, b, npts):
    """Integrate over random points in the triangle."""
    tri = normalized_triangle_pts(a, b)
    bbox = np.array(
        [[tri[:, 0].min(), tri[:, 0].max()], [tri[:, 1].min(), tri[:, 1].max()]]
    )
    scb = bbox[:, 0]
    sca = bbox[:, 1] - scb

    a_box = sca[0] * sca[1]
    a_tri = np.sqrt((1 - a) * (1 - b) * (a + b - 1))  # heron's formula
    npts_comp = int(npts * a_box / a_tri)

    sum = 0.0
    nsum = 0
    for _i in nb.prange(npts_comp):  # try to get the same number of points
        pts = np.random.uniform(0, 1, size=(2, 2))
        pts = pts * sca + scb
        if point_in_triangle(pts[0], tri) and point_in_triangle(pts[1], tri):
            sum += np.log((pts[0, 0] - pts[1, 0]) ** 2 + (pts[0, 1] - pts[1, 1]) ** 2)
            nsum += 1
    sum /= nsum
    return sum / 2, nsum


@nb.njit(parallel=True, cache=True)
def log_gmd_integral_even(a, b, npts):
    """Integrate over random points in the triangle."""
    tri = normalized_triangle_pts(a, b)
    bounds = np.array(
        [[tri[:, 0].min(), tri[:, 0].max()], [tri[:, 1].min(), tri[:, 1].max()]]
    )
    sum = 0.0
    for _i in nb.prange(npts):
        pts = random_pts_in_tri(tri, bounds)
        sum += np.log((pts[0, 0] - pts[1, 0]) ** 2 + (pts[0, 1] - pts[1, 1]) ** 2)
    sum /= npts
    return sum / 2


def generate_valid_triangle():
    """Generate a valid triangle.

    Obviously, 1 side must be at most 1/2 the perimeter, or 1.
    The second side is also maximum on 1, but minimum of 1-side_1
    """
    a = np.random.uniform(0, 1)
    b = np.random.uniform(1 - a, 1)
    return a, b


def plot_pts_in_triangle(a, b, npts):
    """Generate random points in a triangle and plot them.

    Args:
        a (float): length of side a
        b (float): length of side b
        npts (int): number of points to generate
    """
    import matplotlib.pyplot as plt

    tri = normalized_triangle_pts(a, b)
    # print(a, b)
    # print(tri)
    randx = np.random.uniform(tri[:, 0].min(), tri[:, 0].max(), size=(npts, 1))
    randy = np.random.uniform(tri[:, 1].min(), tri[:, 1].max(), size=(npts, 1))
    pts = np.hstack((randx, randy))
    # print(pts.shape)
    inp = points_in_triangle(pts, tri)

    plt.plot(tri[(0, 1, 2, 0), 0], tri[(0, 1, 2, 0), 1], "k-")
    plt.gca().set_aspect("equal")
    plt.plot(pts[inp, 0], pts[inp, 1], "g,")
    plt.plot(pts[~inp, 0], pts[~inp, 1], "r,")

    plt.show()


def plot_gdm_vs_pts():
    """Plot the g.m.d. vs number of points."""
    import matplotlib.pyplot as plt

    val = [
        [
            int(10 ** (n / 2)),
            np.exp(log_gmd_integral(2 / 3, 2 / 3, npts=int(10 ** (n / 2)))) * 3 / 2,
        ]
        for n in range(12, 18)
    ]
    print(val)
    plt.plot(val[:, 0], val[:, 1])
    plt.show()


# @nb.njit(parallel=True)
def generate_fitting_block(bsize, npts):
    """Generate a block of fitting data."""
    res = np.zeros((bsize, 4), dtype=np.float64)
    #   for i in nb.prange(bsize):
    for i in range(bsize):
        a, b = generate_valid_triangle()
        log_r, nsamp = log_gmd_integral(a, b, npts)
        res[i, 0] = a
        res[i, 1] = b
        res[i, 2] = log_r
        res[i, 3] = nsamp
    return res


def generate_fitting_data(fname, fsize, bsize, npts):
    """Generate fitting data."""
    import progressbar

    data = []
    bar = progressbar.ProgressBar(max_value=fsize // bsize)
    for _ in bar(range(fsize // bsize)):
        block = generate_fitting_block(bsize, npts)
        data.append(block)
    np.save(fname, np.vstack(data))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument(
        "--equilateral", "-e", action="store_true", help="g.d.m equilateral triangle"
    )
    options.add_argument(
        "--plot_triangle", "-t", action="store_true", help="plot points in triangle"
    )
    options.add_argument(
        "--plot_integral", "-i", action="store_true", help="plot integral"
    )
    options.add_argument(
        "--generate", "-g", action="store_true", help="generate fitting data"
    )

    fitgroup = options.add_argument_group("generate fitting data")
    fitgroup.add_argument("--fname", type=str, help="file name")
    fitgroup.add_argument("--fsize", type=int, help="file size", default=10000)
    fitgroup.add_argument("--bsize", type=int, help="block size", default=10)

    parser.add_argument(
        "--npts", "-n", type=int, help="number of points", default=1000000
    )

    args = parser.parse_args()
    if args.plot_triangle:
        ab = generate_valid_triangle()
        plot_pts_in_triangle(*ab, npts=args.npts)
    elif args.plot_integral:
        plot_gdm_vs_pts()
    elif args.equilateral:
        # npts = 10000 * 100000
        # Weaver calculates the g.m.d. of an equilateral triangle with sides = 1
        # to be 0.308382 with 10^8 points, i get 0.30838()
        print(
            "g.m.d of equilateral triangle with sides = 1: ",
            np.exp(log_gmd_integral(2 / 3, 2 / 3, npts=args.npts)) * 3 / 2,
        )
    elif args.generate:
        # this currently does 10^7 npts, for 10000 triangles in 9.25 min.
        # weird bit of pauses.
        fname = args.fname if args.fname is not None else "fitting_data.npy"
        generate_fitting_data(fname, args.fsize, args.bsize, args.npts)
