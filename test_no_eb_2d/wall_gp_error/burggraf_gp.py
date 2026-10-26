"""Pressure-gradient error of the Burggraf cavity (benchmark.burggraf, probtype 16) next to the
no-slip walls.  Needs only numpy.

usage: python3 burggraf_gp.py MU PLOTFILE [PLOTFILE ...]

The exact pressure is p = 8 mu (F g''' + f' g') + 64 (f^2/2) (g g'' - g'^2), with
f = x^4 - 2x^3 + x^2, g = y^4 - y^2 and F the integral of f (Shih, Tan & Hwang 1989).  With
incflo's Burggraf forcing it satisfies the steady momentum equations exactly.

For each plotfile it prints, leaving out n/8 cells at each end of every wall (the corners):
  - c on each wall, from a least-squares fit of the wall-normal gp error in the wall cells to
    gp_n - dp/dn = -c mu u_n'', where u_n'' is the second normal derivative of the normal
    velocity at the wall,
  - the largest wall-normal gp error in the wall cells and in the second row of cells,
and the largest gp error in the two cells at the lid corners.
"""
import os
import re
import sys

import numpy as np


def read_level0(path, names):
    """Level-0 data of a cell-centred AMReX plotfile: {name: array[nx, ny]}."""
    with open(os.path.join(path, "Header")) as f:
        lines = f.read().split("\n")
    nvar = int(lines[1])
    varnames = [s.strip() for s in lines[2:2 + nvar]]
    dim = int(lines[2 + nvar])
    assert dim == 2, "2D plotfiles only"
    # The level-0 domain follows prob_lo, prob_hi and the (possibly empty) refinement ratios
    m = re.search(r"\(\(([-\d,]+)\) \(([-\d,]+)\)", lines[7 + nvar] + lines[8 + nvar])
    lo = [int(s) for s in m.group(1).split(",")]
    hi = [int(s) for s in m.group(2).split(",")]
    out = {nm: np.full((hi[0] - lo[0] + 1, hi[1] - lo[1] + 1), np.nan) for nm in names}
    ldir = os.path.join(path, "Level_0")
    with open(os.path.join(ldir, "Cell_H")) as f:
        fods = re.findall(r"FabOnDisk: (\S+) (\d+)", f.read())
    for fname, offset in fods:
        with open(os.path.join(ldir, fname), "rb") as f:
            f.seek(int(offset))
            hdr = f.readline().decode()
            m = re.search(r"\)\)\)\(\(([-\d,]+)\) \(([-\d,]+)\)", hdr)
            blo = [int(s) for s in m.group(1).split(",")]
            bhi = [int(s) for s in m.group(2).split(",")]
            ncomp = int(hdr.split()[-1])
            shape = (bhi[1] - blo[1] + 1, bhi[0] - blo[0] + 1)
            little = re.search(r"\(8, \(([\d ]+)\)\)\)", hdr).group(1).split()[0] == "8"
            data = np.frombuffer(f.read(8 * ncomp * shape[0] * shape[1]),
                                 dtype="<f8" if little else ">f8")
            data = data.reshape((ncomp,) + shape)
        for nm in names:
            out[nm][blo[0]:bhi[0] + 1, blo[1]:bhi[1] + 1] = data[varnames.index(nm)].T
    return out


def exact_gp(x, y, mu):
    f, f1, f2 = x**4 - 2*x**3 + x**2, 4*x**3 - 6*x**2 + 2*x, 12*x**2 - 12*x + 2
    F = x**5/5 - x**4/2 + x**3/3
    g, g1, g2, g3 = y**4 - y**2, 4*y**3 - 2*y, 12*y**2 - 2, 24*y
    px = 8*mu*(f*g3 + f2*g1) + 64*f*f1*(g*g2 - g1**2)
    py = 8*mu*(F*24 + f1*g2) + 32*f**2*(g*g3 - g1*g2)
    return px, py


def main():
    mu = float(sys.argv[1])
    print("    n    c: x walls  bottom    lid    max |gp_n err|: wall cells  2nd row   lid corners")
    for path in sys.argv[2:]:
        d = read_level0(path, ["gpx", "gpy"])
        n = d["gpx"].shape[0]
        s = (np.arange(n) + 0.5) / n
        X, Y = np.meshgrid(s, s, indexing="ij")
        px, py = exact_gp(X, Y, mu)
        ex, ey = d["gpx"] - px, d["gpy"] - py
        t = slice(n//8, n - n//8)  # wall cells away from the corners
        st = s[t]
        # Second normal derivative of the normal velocity at each wall: u_xx = 32 y (2y^2 - 1)
        # at x = 0 and 1, v_yy = 32 x(x-1)(2x-1) at y = 0 and -5 times that at y = 1
        uxx = 32*st*(2*st**2 - 1)
        vyy = 32*st*(st - 1)*(2*st - 1)
        walls = {"x": (np.concatenate([ex[0, t], ex[n-1, t]]), np.concatenate([uxx, uxx])),
                 "bottom": (ey[t, 0], vyy),
                 "lid": (ey[t, n-1], -5*vyy)}
        c = {w: -np.dot(e, mu*r)/np.dot(mu*r, mu*r) if mu > 0 else np.nan
             for w, (e, r) in walls.items()}
        wall = max(abs(ex[[0, n-1], t]).max(), abs(ey[t, [0, n-1]]).max())
        row2 = max(abs(ex[[1, n-2], t]).max(), abs(ey[t, [1, n-2]]).max())
        corner = max(abs(ex[[0, n-1], n-1]).max(), abs(ey[[0, n-1], n-1]).max())
        print("  %3d   %10.3f %7.3f %7.3f   %20.2e %10.2e %11.2e"
              % (n, c["x"], c["bottom"], c["lid"], wall, row2, corner))


if __name__ == "__main__":
    main()
