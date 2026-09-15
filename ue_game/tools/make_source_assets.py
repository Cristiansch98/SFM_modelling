"""Generate the raw source assets the EmvHighway level needs.

Writes (into ue_game/SourceAssets/):
  siren_loop.wav      seamless two-tone emergency siren, 48 kHz 16-bit mono
  SM_NpcCar_*.obj     low-poly NPC cars, +X forward, pivot on the wheel-contact
                      plane, authored in centimetres so UE imports 1:1

Run with plain CPython; no Unreal needed.
"""
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "SourceAssets")

# ----------------------------------------------------------------- siren wav
SR = 48000


def siren_wav(path, period=1.6, f_lo=650.0, f_hi=1350.0):
    """Two-tone wail that loops seamlessly: the phase and the sweep both close
    exactly on the period boundary."""
    n = int(SR * period)
    frames = bytearray()
    phase = 0.0
    for i in range(n):
        t = i / SR
        # smooth wail between the two tones, one full cycle per period
        k = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / period)
        f = f_lo + (f_hi - f_lo) * k
        phase += 2.0 * math.pi * f / SR
        # main tone + an octave and a fifth for the horn-like bite
        s = (0.62 * math.sin(phase)
             + 0.24 * math.sin(2.0 * phase)
             + 0.11 * math.sin(3.0 * phase))
        s *= 0.85
        frames += struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32000))
    data = bytes(frames)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


# ------------------------------------------------------------------ car obj
class Obj:
    def __init__(self):
        self.v = []
        self.groups = {}      # material -> list of faces (tuples of vertex idx)

    def box(self, mat, cx, cy, cz, sx, sy, sz):
        """Axis-aligned box centred at (cx,cy,cz) with full sizes (sx,sy,sz)."""
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        base = len(self.v) + 1
        for dx in (-hx, hx):
            for dy in (-hy, hy):
                for dz in (-hz, hz):
                    self.v.append((cx + dx, cy + dy, cz + dz))
        # local index helper: bit order x,y,z -> 0..7
        def i(x, y, z):
            return base + (x * 4 + y * 2 + z)
        faces = [
            (i(0, 0, 0), i(0, 1, 0), i(0, 1, 1), i(0, 0, 1)),   # -X
            (i(1, 0, 0), i(1, 0, 1), i(1, 1, 1), i(1, 1, 0)),   # +X
            (i(0, 0, 0), i(0, 0, 1), i(1, 0, 1), i(1, 0, 0)),   # -Y
            (i(0, 1, 0), i(1, 1, 0), i(1, 1, 1), i(0, 1, 1)),   # +Y
            (i(0, 0, 0), i(1, 0, 0), i(1, 1, 0), i(0, 1, 0)),   # -Z
            (i(0, 0, 1), i(0, 1, 1), i(1, 1, 1), i(1, 0, 1)),   # +Z
        ]
        self.groups.setdefault(mat, []).extend(faces)

    def wedge(self, mat, x0, x1, y, z0, z1, taper):
        """Box tapered in Y at the +X end (windscreen-ish cabin)."""
        base = len(self.v) + 1
        hy0, hy1 = y / 2.0, y / 2.0 * taper
        self.v += [(x0, -hy0, z0), (x0, hy0, z0), (x1, -hy1, z0), (x1, hy1, z0),
                   (x0, -hy0, z1), (x0, hy0, z1), (x1, -hy1, z1), (x1, hy1, z1)]
        b = base
        faces = [(b + 0, b + 1, b + 3, b + 2), (b + 4, b + 6, b + 7, b + 5),
                 (b + 0, b + 2, b + 6, b + 4), (b + 1, b + 5, b + 7, b + 3),
                 (b + 0, b + 4, b + 5, b + 1), (b + 2, b + 3, b + 7, b + 6)]
        self.groups.setdefault(mat, []).extend(faces)

    def wheel(self, mat, cx, cy, r, w):
        """Low-poly (8-gon) wheel, axis along Y, resting on z = 0."""
        base = len(self.v) + 1
        n = 8
        for side in (-w / 2.0, w / 2.0):
            for k in range(n):
                a = 2.0 * math.pi * k / n
                self.v.append((cx + r * math.cos(a), cy + side,
                               r + r * math.sin(a)))
        f = self.groups.setdefault(mat, [])
        for k in range(n):
            k2 = (k + 1) % n
            f.append((base + k, base + k2, base + n + k2, base + n + k))
        f.append(tuple(base + k for k in range(n)))
        f.append(tuple(base + n + k for k in reversed(range(n))))

    def _face_uvs(self, face):
        """Planar-project the face onto whichever axis pair it faces, so the
        importer gets a sane (and non-degenerate) UV set."""
        pts = [self.v[i - 1] for i in face]
        ax, ay, az = (max(p[k] for p in pts) - min(p[k] for p in pts)
                      for k in range(3))
        if az <= ax and az <= ay:
            pick = (0, 1)
        elif ay <= ax:
            pick = (0, 2)
        else:
            pick = (1, 2)
        return [(pts[i][pick[0]] / 100.0, pts[i][pick[1]] / 100.0)
                for i in range(len(pts))]

    def write(self, path, name):
        # companion .mtl -> one UE material slot per usemtl group
        mtl_path = os.path.splitext(path)[0] + ".mtl"
        with open(mtl_path, "w") as mh:
            for m in self.groups:
                mh.write("newmtl %s\nKd 0.5 0.5 0.5\n\n" % m)
        with open(path, "w") as fh:
            fh.write("# %s - generated for EmvHighway\n" % name)
            fh.write("mtllib %s\n" % os.path.basename(mtl_path))
            fh.write("o %s\n" % name)
            for x, y, z in self.v:
                # The OBJ importer maps (x,y,z) -> UE (x,-y,z), so flip Y here
                # to end up with UE's +X forward / +Y right / +Z up.
                fh.write("v %.2f %.2f %.2f\n" % (x, -y, z))
            uv_lines = []
            face_lines = []
            for mat, faces in self.groups.items():
                face_lines.append("usemtl %s" % mat)
                for f in faces:
                    corners = []
                    for (u, v) in self._face_uvs(f):
                        uv_lines.append("vt %.4f %.4f" % (u, v))
                        corners.append(len(uv_lines))
                    face_lines.append("f " + " ".join(
                        "%d/%d" % (vi, ti) for vi, ti in zip(f, corners)))
            fh.write("\n".join(uv_lines) + "\n")
            fh.write("\n".join(face_lines) + "\n")


def car(length, width, roof, cab_front, cab_back, cab_taper=0.86,
        ride=None, wheel_r=32.0, wheel_w=22.0, box_body=False):
    """Build a generic car. All dimensions in centimetres, +X forward."""
    o = Obj()
    sill = ride if ride is not None else wheel_r * 0.55
    body_top = roof * 0.62 if not box_body else roof * 0.55
    # main body
    o.box("Body", 0.0, 0.0, (sill + body_top) / 2.0,
          length, width, body_top - sill)
    # cabin / greenhouse (glass + pillars)
    if box_body:
        o.box("Trim", (cab_back + cab_front) / 2.0, 0.0,
              (body_top + roof) / 2.0, cab_front - cab_back, width * 0.98,
              roof - body_top)
    else:
        o.wedge("Trim", cab_back, cab_front, width * 0.92, body_top, roof,
                cab_taper)
    # bumpers
    o.box("Trim", length / 2.0 - 6.0, 0.0, sill + 12.0, 14.0, width * 0.96, 20.0)
    o.box("Trim", -length / 2.0 + 6.0, 0.0, sill + 12.0, 14.0, width * 0.96, 20.0)
    # wheels
    ax_f = length * 0.31
    ax_r = -length * 0.31
    for ax in (ax_f, ax_r):
        for sy in (-1.0, 1.0):
            o.wheel("Trim", ax, sy * (width / 2.0 - wheel_w / 2.0 - 1.0),
                    wheel_r, wheel_w)
    return o


CARS = {
    # name          length width roof  cabF   cabB  extra
    "SM_NpcCar_Sedan":   dict(length=450, width=180, roof=145, cab_front=90,
                              cab_back=-110),
    "SM_NpcCar_Hatch":   dict(length=398, width=176, roof=152, cab_front=70,
                              cab_back=-120, wheel_r=30.0),
    "SM_NpcCar_Wagon":   dict(length=478, width=182, roof=155, cab_front=95,
                              cab_back=-180),
    "SM_NpcCar_Van":     dict(length=540, width=200, roof=215, cab_front=150,
                              cab_back=-190, box_body=True, wheel_r=36.0,
                              wheel_w=26.0),
    "SM_NpcCar_Pickup":  dict(length=560, width=200, roof=180, cab_front=40,
                              cab_back=-90, wheel_r=38.0, wheel_w=28.0),
    "SM_NpcCar_City":    dict(length=362, width=170, roof=150, cab_front=60,
                              cab_back=-105, wheel_r=28.0),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    siren_wav(os.path.join(OUT, "siren_loop.wav"))
    for name, kw in CARS.items():
        car(**kw).write(os.path.join(OUT, name + ".obj"), name)
    print("wrote", len(CARS) + 1, "files to", OUT)


if __name__ == "__main__":
    main()
