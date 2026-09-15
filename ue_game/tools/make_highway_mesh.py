"""Generate SM_Highway.obj - the whole motorway as one low-poly mesh.

Coordinate contract (see docs/UE_BRIDGE.md), all centimetres:
    road runs along +X from 0 to 300000 (3 km)
    middle lane centred on Y = 0, lanes 350 wide
    lane centres: right +350, middle 0, left -350
    carriageway edges at Y = +-525, road surface at Z = 0

Material groups: Asphalt / Marking / Rail / Grass  ->  4 UE material slots.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "SourceAssets", "SM_Highway.obj")

LENGTH = 300000.0       # 3 km
X0, X1 = 0.0, LENGTH
EDGE = 525.0            # carriageway edge (3 x 3.5 m lanes)
SHOULDER = 700.0        # hard-shoulder outer edge
LANE_LINE = 175.0       # dashed lane boundaries
LINE_W = 16.0           # painted line width
SLAB_BOTTOM = -35.0
MARK_Z = 0.6            # markings float just above the asphalt
DASH, GAP = 300.0, 900.0

RAIL_Y = 760.0
RAIL_Z0, RAIL_Z1 = 45.0, 78.0
RAIL_T = 9.0
POST_EVERY = 800.0
POST_W, POST_Z = 14.0, 62.0

VERGE_Y, VERGE_Z = 1500.0, -190.0
GROUND_Y, GROUND_X_PAD = 90000.0, 40000.0


class Obj:
    def __init__(self):
        self.v = []
        self.vt = []
        self.f = []          # (material, [(vi, ti), ...])

    def quad(self, mat, pts, uv_scale=0.01):
        """pts: 4 (x,y,z) in winding order, CCW seen from outside."""
        base = len(self.v) + 1
        self.v.extend(pts)
        # planar UV: drop the axis with the smallest extent
        ext = [max(p[k] for p in pts) - min(p[k] for p in pts) for k in range(3)]
        drop = ext.index(min(ext))
        pick = [k for k in range(3) if k != drop]
        tbase = len(self.vt) + 1
        for p in pts:
            self.vt.append((p[pick[0]] * uv_scale, p[pick[1]] * uv_scale))
        self.f.append((mat, [(base + i, tbase + i) for i in range(4)]))

    def box(self, mat, x0, x1, y0, y1, z0, z1):
        self.quad(mat, [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)])
        self.quad(mat, [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)])
        self.quad(mat, [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)])
        self.quad(mat, [(x1, y1, z0), (x1, y1, z1), (x0, y1, z1), (x0, y1, z0)])
        self.quad(mat, [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)])
        self.quad(mat, [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)])

    def top_quad(self, mat, x0, x1, ya, yb, za, zb):
        """Upward-facing quad spanning x0..x1 across ya..yb. Corners are always
        emitted in ascending-Y order, which is the winding that comes out
        facing up once write() flips Y into UE space."""
        (ylo, zlo), (yhi, zhi) = sorted([(ya, za), (yb, zb)])
        self.quad(mat, [(x0, ylo, zlo), (x0, yhi, zhi),
                        (x1, yhi, zhi), (x1, ylo, zlo)])

    def strip_x(self, mat, y0, y1, z, x0=X0, x1=X1, step=5000.0, z1=None):
        """A long flat (or laterally sloping) band along X, split into segments
        so vertices stay reasonably dense for lighting/LOD."""
        x = x0
        while x < x1:
            xe = min(x + step, x1)
            self.top_quad(mat, x, xe, y0, y1, z, z if z1 is None else z1)
            x = xe

    def write(self, path, name):
        # a companion .mtl is what makes the importer create one UE material
        # slot per usemtl group instead of collapsing everything into one
        mats = []
        for mat, _ in self.f:
            if mat not in mats:
                mats.append(mat)
        mtl_path = os.path.splitext(path)[0] + ".mtl"
        with open(mtl_path, "w") as mh:
            for m in mats:
                mh.write("newmtl %s\nKd 0.5 0.5 0.5\n\n" % m)
        with open(path, "w") as fh:
            fh.write("# %s - EmvHighway, 3 km 3-lane motorway along +X\n" % name)
            fh.write("mtllib %s\n" % os.path.basename(mtl_path))
            fh.write("o %s\n" % name)
            for x, y, z in self.v:
                fh.write("v %.2f %.2f %.2f\n" % (x, -y, z))   # OBJ -> UE Y flip
            for u, t in self.vt:
                fh.write("vt %.4f %.4f\n" % (u, t))
            cur = None
            for mat, corners in self.f:
                if mat != cur:
                    fh.write("usemtl %s\n" % mat)
                    cur = mat
                fh.write("f " + " ".join("%d/%d" % c for c in corners) + "\n")


def build():
    o = Obj()

    # ---- road slab: asphalt top, edges, underside -------------------------
    o.strip_x("Asphalt", -SHOULDER, SHOULDER, 0.0)
    o.strip_x("Asphalt", -SHOULDER, SHOULDER, SLAB_BOTTOM, step=LENGTH)
    for sy in (-1.0, 1.0):
        y = sy * SHOULDER
        pts = [(X0, y, SLAB_BOTTOM), (X0, y, 0.0), (X1, y, 0.0), (X1, y, SLAB_BOTTOM)]
        o.quad("Asphalt", pts if sy > 0 else list(reversed(pts)))

    # ---- painted markings -------------------------------------------------
    for sy in (-1.0, 1.0):                       # solid carriageway edges
        c = sy * EDGE
        o.strip_x("Marking", c - LINE_W / 2, c + LINE_W / 2, MARK_Z, step=20000.0)
    for sy in (-1.0, 1.0):                       # dashed lane boundaries
        c = sy * LANE_LINE
        x = X0
        while x < X1:
            xe = min(x + DASH, X1)
            o.top_quad("Marking", x, xe, c - LINE_W / 2, c + LINE_W / 2,
                       MARK_Z, MARK_Z)
            x += DASH + GAP

    # ---- guardrails -------------------------------------------------------
    for sy in (-1.0, 1.0):
        y = sy * RAIL_Y
        seg = 10000.0
        x = X0
        while x < X1:
            xe = min(x + seg, X1)
            o.box("Rail", x, xe, y - RAIL_T / 2, y + RAIL_T / 2, RAIL_Z0, RAIL_Z1)
            x = xe
        x = X0
        while x < X1:
            o.box("Rail", x - POST_W / 2, x + POST_W / 2,
                  y - POST_W / 2, y + POST_W / 2, 0.0, POST_Z)
            x += POST_EVERY

    # ---- verges + ground --------------------------------------------------
    for sy in (-1.0, 1.0):
        o.strip_x("Grass", sy * SHOULDER, sy * VERGE_Y, 0.0, step=10000.0,
                  z1=VERGE_Z)
        o.strip_x("Grass", sy * VERGE_Y, sy * GROUND_Y, VERGE_Z,
                  X0 - GROUND_X_PAD, X1 + GROUND_X_PAD, step=50000.0)
    # ground caps beyond the ends of the road
    for x0, x1 in ((X0 - GROUND_X_PAD, X0), (X1, X1 + GROUND_X_PAD)):
        o.top_quad("Grass", x0, x1, -VERGE_Y, VERGE_Y, VERGE_Z, VERGE_Z)
    return o


if __name__ == "__main__":
    obj = build()
    obj.write(OUT, "SM_Highway")
    print("wrote %s: %d verts, %d faces" % (OUT, len(obj.v), len(obj.f)))
