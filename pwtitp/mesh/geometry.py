"""Geometric acoustics in furnished 3D-FRONT houses (GPU).

Image sources of an arbitrary triangle mesh: candidate reflection sequences are found by specular ray tracing from the
source (as in GSound-type hybrid simulators) and every candidate is validated deterministically for each receiver
position (back-tracing through the reflecting polygons and occlusion test against the whole mesh), so that a reflection
exists exactly where it is geometrically visible and its delay drifts continuously with the receiver. Per-face
materials: octave-band absorption a and scattering s (sound_a / sound_s in the .mtl); a specular reflection scales the
band amplitude by sqrt((1-a)(1-s)); air absorption per band and path length. The scattered field is estimated by
diffuse-rain ray tracing (energy and energy-weighted arrival direction per band and time bin).
Ray queries use NVIDIA Warp if available, otherwise a brute-force Moller-Trumbore fallback in torch."""
import math, os, re
import sys
import numpy as np
import torch
if os.environ.get('PWTITP_PYLIBS'):
    sys.path.insert(0, os.environ['PWTITP_PYLIBS'])
try:
    import warp as wp
    wp.config.quiet = True
    wp.init()

    @wp.kernel
    def _wp_ray(mesh: wp.uint64, o: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3),
                tout: wp.array(dtype=float), fout: wp.array(dtype=int)):
        i = wp.tid()
        dd = d[i]
        L = wp.length(dd)
        if L < 1.0e-12:
            tout[i] = 1.0e30
            fout[i] = -1
            return
        q = wp.mesh_query_ray(mesh, o[i], dd / L, 1.0e6)
        if q.result:
            tout[i] = q.t / L
            fout[i] = q.face
        else:
            tout[i] = 1.0e30
            fout[i] = -1

    @wp.kernel
    def _wp_validate(mesh: wp.uint64, fplane: wp.array(dtype=int), pn: wp.array(dtype=wp.vec3), pgid: wp.array(dtype=int),
                     chain: wp.array2d(dtype=int), nlen: wp.array(dtype=int), imgs: wp.array2d(dtype=wp.vec3),
                     src: wp.vec3, pos: wp.array(dtype=wp.vec3), pc: wp.array(dtype=int), pp: wp.array(dtype=int),
                     eps: float, out: wp.array(dtype=int)):
        """one thread per (chain, receiver) pair: back-trace receiver -> I_n -> ... -> I_1 -> src (same rules as
        MeshRoom.validate_pairs)."""
        i = wp.tid()
        c = pc[i]
        o = wp.vec3(pos[pp[i]])
        n = nlen[c]
        for jj in range(n):
            j = n - 1 - jj
            pl = chain[c, j]
            dv = imgs[c, j] - o
            L = wp.length(dv)
            if L < 1.0e-9:
                out[i] = 0
                return
            dn = dv / L
            q = wp.mesh_query_ray(mesh, o, dn, L)
            if not q.result:
                out[i] = 0
                return
            hit = o + dn * q.t
            nrm = pn[pl]
            if pgid[fplane[q.face]] != pgid[pl]:  # must hit the expected reflector (or a duplicate copy of it)
                out[i] = 0
                return
            sd = wp.sign(wp.dot(o - hit, nrm))
            o = hit + nrm * (sd * eps)
        dv = src - o
        L = wp.length(dv)
        dn = dv / wp.max(L, 1.0e-9)
        o2 = o + dn * eps
        if L > 2.0 * eps:
            q = wp.mesh_query_ray(mesh, o2, dn, L - 2.0 * eps)
            if q.result:
                out[i] = 0
                return
        out[i] = 1
except Exception:  # pragma: no cover
    wp = None

BANDS = np.array([62.5, 125, 250, 500, 1000, 2000, 4000, 8000])
AIR_DB_M = np.array([0.0001, 0.0004, 0.0011, 0.0019, 0.0037, 0.0097, 0.0329, 0.1170])  # ISO 9613-1, 20 C, 50 % RH
C = 343.0


# ----------------------------------------------------------------------------------------------------------- loading
def _parse_mtl(path):
    mats, cur = {}, None
    for ln in open(path, errors='ignore'):
        t = ln.split()
        if not t:
            continue
        if t[0] == 'newmtl':
            cur = ' '.join(t[1:]); mats[cur] = {}
        elif t[0] in ('sound_a', 'sound_s') and cur is not None:
            mats[cur][t[0]] = np.array([float(x) for x in t[1:9]])
    return mats


def load_obj(path):
    """triangles [F,3,3] float64, material index per face, material table (a, s: [M,8]). Polygons are fan-triangulated."""
    verts, faces, fm = [], [], []
    names, cur = [], -1
    mtl = None
    with open(path, errors='ignore') as f:
        for ln in f:
            if ln.startswith('v '):
                verts.append(ln.split()[1:4])
            elif ln.startswith('f '):
                idx = [int(p.split('/')[0]) for p in ln.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1])); fm.append(cur)
            elif ln.startswith('usemtl'):
                nm = ln.split(maxsplit=1)[1].strip() if len(ln.split()) > 1 else ''
                if nm not in names:
                    names.append(nm)
                cur = names.index(nm)
            elif ln.startswith('mtllib'):
                mtl = ln.split(maxsplit=1)[1].strip()
    V = np.asarray(verts, dtype=np.float64)
    Fi = np.asarray(faces, dtype=np.int64)
    Fi = np.where(Fi > 0, Fi - 1, len(V) + Fi)
    mats = _parse_mtl(os.path.join(os.path.dirname(path), mtl)) if mtl else {}
    a = np.full((max(len(names), 1), 8), np.nan); s = np.full_like(a, np.nan)
    for i, nm in enumerate(names):
        if nm in mats and 'sound_a' in mats[nm] and 'sound_s' in mats[nm]:
            a[i] = mats[nm]['sound_a']; s[i] = mats[nm]['sound_s']
    return V[Fi], np.asarray(fm), names, a, s


# ----------------------------------------------------------------------------------------------------------- scene
class MeshRoom:
    def __init__(self, obj_path, device='cuda', drop=('bounds',)):
        tri, fm, names, a, s = load_obj(obj_path)
        keep = np.array([fm_i < 0 or names[fm_i] not in drop for fm_i in fm])
        tri, fm = tri[keep], fm[keep]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(n, axis=1)
        ok = area2 > 1e-8
        tri, fm, n, area2 = tri[ok], fm[ok], n[ok], area2[ok]
        n = n / area2[:, None]
        self.used_materials = sorted({names[i] for i in set(fm.tolist()) if i >= 0})
        self.colored = bool(len(fm) and (fm >= 0).all() and np.isfinite(a[np.unique(fm)]).all() and np.isfinite(s[np.unique(fm)]).all())
        self.tri_np, self.area = tri, area2 / 2
        self.bounds = np.stack([tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)])
        # planes: unoriented normal + offset (1 cm / ~0.1 deg) + material -> one reflector id
        sgn = np.sign(n[np.arange(len(n)), np.argmax(np.abs(n), 1)])
        nu = n * sgn[:, None]
        d = (nu * tri[:, 0]).sum(1)
        key = np.concatenate([np.round(nu / 2e-3), np.round(d / 0.01)[:, None], fm[:, None]], 1).astype(np.int64)
        _, pid = np.unique(key, axis=0, return_inverse=True)
        pid = pid.reshape(-1)
        P = pid.max() + 1
        pn = np.zeros((P, 3)); pd = np.zeros(P); pw = np.zeros(P)
        np.add.at(pn, pid, nu * self.area[:, None]); np.add.at(pd, pid, d * self.area); np.add.at(pw, pid, self.area)
        pn /= np.linalg.norm(pn, axis=1, keepdims=True); pd /= pw
        pmat = np.zeros(P, dtype=np.int64); pmat[pid] = fm
        self.P = P
        dev = self.dev = device
        T = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=dev)
        self.v0 = T(tri[:, 0]); self.e1 = T(tri[:, 1] - tri[:, 0]); self.e2 = T(tri[:, 2] - tri[:, 0])
        self.fplane = T(pid, torch.long)
        self.pn = T(pn, torch.float64); self.pd = T(pd, torch.float64)
        aa = np.nan_to_num(a, nan=0.1); ss = np.nan_to_num(s, nan=0.1)
        self.mat_a = T(aa, torch.float64); self.mat_s = T(ss, torch.float64)
        self.pmat = T(pmat, torch.long)
        # near-coincident duplicate surfaces (3D-FRONT has copies <= 1 cm apart, parallel within ~1 deg): one acoustic
        # reflector, so a ray path is not discovered / counted twice (gid used for chain de-duplication and validation)
        dot = np.abs(pn @ pn.T); dgap = np.abs(pd[:, None] - np.sign(pn @ pn.T) * pd[None])
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        gi, gj = np.nonzero((dot > 0.999) & (dgap < 0.01))
        gid = connected_components(coo_matrix((np.ones(len(gi)), (gi, gj)), shape=(P, P)), directed=False)[1]
        self.pgid = T(gid, torch.long); self.face_gid = self.pgid[self.fplane]
        # specular band amplitude per reflector
        self.prefl = torch.sqrt((1 - self.mat_a[self.pmat]).clamp(0, 1) * (1 - self.mat_s[self.pmat]).clamp(0, 1))  # [P,8]
        self.face_a = T(aa[fm], torch.float64); self.face_s = T(ss[fm], torch.float64)
        self.wp_mesh = None
        if wp is not None and str(dev).startswith('cuda'):
            self._wp_dev = 'cuda:%d' % (torch.device(dev).index or 0)
            self._wp_pts = wp.array(tri.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=self._wp_dev)
            self._wp_idx = wp.array(np.arange(3 * len(tri), dtype=np.int32), dtype=int, device=self._wp_dev)
            self.wp_mesh = wp.Mesh(points=self._wp_pts, indices=self._wp_idx)

    # ---- ray queries
    @torch.no_grad()
    def nearest_hit(self, o, d, chunk=1 << 22):
        """o, d [R,3] (d need not be unit). returns t [R] (ray parameter, inf if no hit, t>1e-6) and face [R] (-1)."""
        R, F = o.shape[0], self.v0.shape[0]
        if self.wp_mesh is not None and not getattr(self, 'force_brute', False):
            if R == 0:
                return torch.zeros(0, device=self.dev), torch.zeros(0, dtype=torch.long, device=self.dev)
            oo = o.float().contiguous(); dd = d.float().contiguous()
            tout = torch.empty(R, device=self.dev); fo = torch.empty(R, dtype=torch.int32, device=self.dev)
            st = wp.stream_from_torch(torch.cuda.current_stream(self.dev))
            wp.launch(_wp_ray, dim=R, inputs=[self.wp_mesh.id, wp.from_torch(oo, dtype=wp.vec3), wp.from_torch(dd, dtype=wp.vec3),
                                               wp.from_torch(tout), wp.from_torch(fo)], device=self._wp_dev, stream=st)
            fo = fo.long()
            hit = (fo >= 0) & (tout > 1e-6)
            return torch.where(hit, tout, torch.full_like(tout, float('inf'))), torch.where(hit, fo, torch.full_like(fo, -1))
        rb = max(1, chunk // F)
        tout = torch.full((R,), float('inf'), device=self.dev); fout = torch.full((R,), -1, dtype=torch.long, device=self.dev)
        e1, e2, v0 = self.e1[None], self.e2[None], self.v0[None]
        for i in range(0, R, rb):
            oo, dd = o[i:i + rb, None].float(), d[i:i + rb, None].float()
            p = torch.cross(dd.expand(-1, F, -1), e2.expand(dd.shape[0], -1, -1), dim=-1)
            det = (e1 * p).sum(-1)
            inv = 1.0 / torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
            tv = oo - v0
            u = (tv * p).sum(-1) * inv
            q = torch.cross(tv, e1.expand(dd.shape[0], -1, -1), dim=-1)
            v = (dd * q).sum(-1) * inv
            t = (e2 * q).sum(-1) * inv
            hit = (det.abs() > 1e-12) & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-6)
            t = torch.where(hit, t, torch.full_like(t, float('inf')))
            tm, fi = t.min(1)
            tout[i:i + rb] = tm; fout[i:i + rb] = torch.where(torch.isfinite(tm), fi, torch.full_like(fi, -1))
        return tout, fout

    def segment_blocked(self, a, b, eps=1e-3):
        """True where the open segment a->b [R,3] hits the mesh (ignoring eps metres at both ends)."""
        d = b - a
        L = d.norm(dim=-1).clamp_min(1e-9)
        o = a + d * (eps / L)[:, None]
        t, _ = self.nearest_hit(o, d)
        return t < (1 - 2 * eps / L)

    # ---- image sources
    def reflect(self, x, pids):
        """mirror points x [N,3] (float64) across reflectors pids [N]."""
        n = self.pn[pids]; dist = (x * n).sum(-1) - self.pd[pids]
        return x - 2 * dist[:, None] * n

    @torch.no_grad()
    def discover(self, src, n_rays=40000, max_order=4, seed=0):
        """specular ray tracing from src: unique reflector sequences up to max_order + room statistics
        (mean free path, hit-weighted band absorption / scattering, escape fraction)."""
        g = torch.Generator(device=self.dev).manual_seed(seed)
        d = torch.randn(n_rays, 3, device=self.dev, generator=g, dtype=torch.float64); d = d / d.norm(dim=-1, keepdim=True)
        o = torch.as_tensor(src, dtype=torch.float64, device=self.dev).expand(n_rays, 3).clone()
        seqs = torch.full((n_rays, max_order), -1, dtype=torch.long, device=self.dev)
        alive = torch.ones(n_rays, dtype=torch.bool, device=self.dev)
        seglen, hits_a, hits_s, nhit = [], [], [], 0
        escaped = 0
        for k in range(max(max_order, 6)):
            idx = alive.nonzero().squeeze(1)
            if len(idx) == 0:
                break
            t, f = self.nearest_hit(o[idx], d[idx])
            ok = f >= 0
            if k == 0:
                escaped = int((~ok).sum())
            alive[idx[~ok]] = False
            idx, t, f = idx[ok], t[ok].double(), f[ok]
            seglen.append(t); hits_a.append(self.face_a[f]); hits_s.append(self.face_s[f])
            if k < max_order:
                seqs[idx, k] = self.fplane[f]
            x = o[idx] + d[idx] * t[:, None]
            nrm = self.pn[self.fplane[f]]
            dd = d[idx] - 2 * (d[idx] * nrm).sum(-1, keepdim=True) * nrm
            o[idx] = x + dd * 1e-4; d[idx] = dd
        mfp = float(torch.cat(seglen[1:]).mean()) if len(seglen) > 1 else float('nan')  # skip source->first wall leg
        st = dict(mfp=mfp, a=torch.cat(hits_a).mean(0).cpu().numpy(), s=torch.cat(hits_s).mean(0).cpu().numpy(),
                  escape=escaped / n_rays)
        chains = []
        for n in range(1, max_order + 1):
            u = torch.unique(seqs[:, :n][(seqs[:, :n] >= 0).all(1)], dim=0)
            u = u[(u[:, 1:] != u[:, :-1]).all(1)] if n > 1 else u
            chains.append(u)
        return chains, st

    def images(self, src, chain):
        """image positions [C, n, 3] (I_1..I_n) of src for chains [C, n]"""
        C_, n = chain.shape
        x = torch.as_tensor(src, dtype=torch.float64, device=self.dev).expand(C_, 3)
        out = []
        for j in range(n):
            x = self.reflect(x, chain[:, j]); out.append(x)
        return torch.stack(out, 1)

    @torch.no_grad()
    def validate(self, src, chain, imgs, pos, eps=1e-4):
        """visibility of each image path at each receiver position: [C, J] bool (all chain x position pairs)."""
        C_, n = chain.shape; J = pos.shape[0]
        return self.validate_pairs(src, chain.repeat_interleave(J, 0), imgs.repeat_interleave(J, 0),
                                   pos.double().repeat(C_, 1), eps).view(C_, J)

    @torch.no_grad()
    def validate_pairs(self, src, chain, imgs, pos, eps=1e-4):
        """visibility of image path chain[r] at receiver pos[r], r = 0..R-1: [R] bool. Back-traces receiver -> I_n -> ...
        -> src; each leg's first hit must be a face of the expected reflector, and the last leg must reach the source
        unblocked."""
        R, n = chain.shape
        src_t = torch.as_tensor(src, dtype=torch.float64, device=self.dev)
        o = pos.double().clone()
        ok = torch.ones(R, dtype=torch.bool, device=self.dev)
        for j in range(n - 1, -1, -1):
            idx = ok.nonzero().squeeze(1)
            if len(idx) == 0:
                break
            pl = chain[idx, j]
            dvec = imgs[idx, j] - o[idx]
            t, f = self.nearest_hit(o[idx], dvec)
            good = (f >= 0) & (t < 1)
            hitp = o[idx] + dvec * t.double().clamp(max=1)[:, None]
            g2 = good.clone(); g2[good] = self.face_gid[f[good]] == self.pgid[pl[good]]  # expected reflector or a copy
            good = g2
            ok[idx[~good]] = False
            idx, hitp, pl = idx[good], hitp[good], pl[good]
            nrm = self.pn[pl]
            side = torch.sign(((o[idx] - hitp) * nrm).sum(-1, keepdim=True))  # step back towards the incoming side
            o[idx] = hitp + side * nrm * eps
        idx = ok.nonzero().squeeze(1)
        if len(idx):
            blk = self.segment_blocked(o[idx], src_t.expand(len(idx), 3), eps=eps)
            ok[idx[blk]] = False
        return ok

    @torch.no_grad()
    def validate_padded(self, src, chain, nlen, imgs, pos, pc, pp, eps=1e-4, chunk=1 << 23):
        """warp version of validate_pairs for chains of mixed order. chain [C, NMAX] long (padded), nlen [C], imgs
        [C, NMAX, 3], pos [P, 3]; pairs (pc[r], pp[r]) -> [R] bool. Falls back to the torch path without warp."""
        R = pc.shape[0]
        if R == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.dev)
        if self.wp_mesh is None or getattr(self, 'force_brute', False):
            out = torch.zeros(R, dtype=torch.bool, device=self.dev)
            for n in torch.unique(nlen[pc]).tolist():
                m = (nlen[pc] == n).nonzero().squeeze(1)
                out[m] = self.validate_pairs(src, chain[pc[m], :n], imgs[pc[m], :n], pos[pp[m]], eps)
            return out
        if not hasattr(self, '_wp_fplane'):
            self._wp_fplane_t = self.fplane.int().contiguous(); self._wp_pn_t = self.pn.float().contiguous()
            self._wp_pd_t = self.pgid.int().contiguous()
            self._wp_fplane = wp.from_torch(self._wp_fplane_t, dtype=wp.int32); self._wp_pn = wp.from_torch(self._wp_pn_t, dtype=wp.vec3)
            self._wp_pd = wp.from_torch(self._wp_pd_t, dtype=wp.int32)
        ch = chain.int().contiguous(); nl = nlen.int().contiguous(); im = imgs.float().contiguous(); ps = pos.float().contiguous()
        s3 = wp.vec3(*[float(x) for x in np.asarray(torch.as_tensor(src).cpu(), dtype=np.float64)])
        st = wp.stream_from_torch(torch.cuda.current_stream(self.dev))
        out = torch.empty(R, dtype=torch.int32, device=self.dev)
        for i in range(0, R, chunk):
            a, b = pc[i:i + chunk].int().contiguous(), pp[i:i + chunk].int().contiguous()
            o = out[i:i + chunk]
            wp.launch(_wp_validate, dim=a.shape[0], inputs=[self.wp_mesh.id, self._wp_fplane, self._wp_pn, self._wp_pd,
                      wp.from_torch(ch, dtype=wp.int32), wp.from_torch(nl, dtype=wp.int32), wp.from_torch(im.view(im.shape[0], -1, 3), dtype=wp.vec3),
                      s3, wp.from_torch(ps, dtype=wp.vec3), wp.from_torch(a, dtype=wp.int32), wp.from_torch(b, dtype=wp.int32), float(eps),
                      wp.from_torch(o, dtype=wp.int32)], device=self._wp_dev, stream=st)
        return out.bool()

    def chain_gain(self, chain):
        """band amplitude of each chain [C, 8]"""
        return self.prefl[chain].prod(1)

    def eyring_t60(self, st):
        """band T60 (s) from mean free path l = 4V/S and hit-weighted absorption: T = 0.161 (l/4) / (-ln(1-a) + m l)"""
        m = AIR_DB_M / (10 * np.log10(np.e))  # amplitude dB/m -> energy Np/m
        aeff = np.clip(st['a'], 1e-3, 0.95)
        return 0.161 * (st['mfp'] / 4) / (-np.log(1 - aeff) + m * st['mfp'])

    # ---- sampling helpers
    def winding(self, pts):
        import igl
        return igl.fast_winding_number(self.tri_np.reshape(-1, 3), np.arange(3 * len(self.tri_np)).reshape(-1, 3), np.asarray(pts, dtype=np.float64))

    def clearance(self, pts):
        import igl
        V = self.tri_np.reshape(-1, 3); Fi = np.arange(len(V)).reshape(-1, 3)
        d2 = igl.point_mesh_squared_distance(np.asarray(pts, dtype=np.float64), V, Fi)[0]
        return np.sqrt(d2)


def _seg_point_dist(o, d, t, P):
    """closest approach of segments o + u d, u in [0, t] (d unit) [n,3] to points P [R,3]: (dist [n,R], u [n,R])"""
    w = P[None] - o[:, None]                                   # [n,R,3]
    u = (w * d[:, None]).sum(-1).clamp_min(0)
    u = torch.minimum(u, t[:, None])
    q = o[:, None] + d[:, None] * u[..., None]
    return (q - P[None]).norm(dim=-1), u


@torch.no_grad()
def discover_tube(room, src, rpts, n_rays=200000, max_order=50, t_max=0.105, tube=2.0, seed=0, chunk=20000):
    """GSound-style specular path discovery up to max_order reflections: specular rays from src; the reflector sequence
    after k reflections becomes an image-source candidate if the k-th outgoing ray passes within the beam radius
    r = tube * l * sqrt(4 pi / n_rays) + 3 cm of a receiver sample point (l = travelled length). Rays stop at path length
    C*t_max (+ receiver extent). Every candidate is validated later exactly. Also returns room statistics
    (mean free path, hit-weighted band a / s, escape fraction of the first leg). chains: list (order 1..max_order) of
    [C_n, n] long."""
    dev = room.dev
    rpts = torch.as_tensor(rpts, dtype=torch.float64, device=dev)
    ext = float((rpts - rpts.mean(0)).norm(dim=-1).max())
    Lmax = C * t_max + ext
    g = torch.Generator(device=dev).manual_seed(seed)
    d = torch.randn(n_rays, 3, device=dev, generator=g, dtype=torch.float64); d = d / d.norm(dim=-1, keepdim=True)
    o = torch.as_tensor(src, dtype=torch.float64, device=dev).expand(n_rays, 3).clone()
    lacc = torch.zeros(n_rays, dtype=torch.float64, device=dev)
    seqs = torch.full((n_rays, max_order), -1, dtype=torch.long, device=dev)
    alive = torch.ones(n_rays, dtype=torch.bool, device=dev)
    cand = [[] for _ in range(max_order)]
    seglen, hits_a, hits_s, escaped = [], [], [], 0
    ang = math.sqrt(4 * math.pi / n_rays)
    for k in range(max_order + 1):
        idx = alive.nonzero().squeeze(1)
        if len(idx) == 0:
            break
        t, f = room.nearest_hit(o[idx], d[idx])
        ok = f >= 0
        if k == 0:
            escaped = int((~ok).sum())
        t = torch.where(ok, t.double(), torch.full_like(t.double(), 1e3))
        if k >= 1:  # outgoing ray after k reflections: near the receivers?
            for i in range(0, len(idx), chunk):
                ii = idx[i:i + chunk]
                dist, u = _seg_point_dist(o[ii], d[ii], t[i:i + chunk], rpts)
                r = tube * (lacc[ii, None] + u) * ang + 0.03
                near = (dist < r).any(1)
                if near.any():
                    cand[k - 1].append(seqs[ii[near], :k])
        alive[idx[~ok]] = False
        idx, t, f = idx[ok], t[ok], f[ok]
        if k == max_order:
            break
        if k >= 1:
            seglen.append(t)
        hits_a.append(room.face_a[f]); hits_s.append(room.face_s[f])
        seqs[idx, k] = room.fplane[f]
        x = o[idx] + d[idx] * t[:, None]
        lacc[idx] += t
        nrm = room.pn[room.fplane[f]]
        dd = d[idx] - 2 * (d[idx] * nrm).sum(-1, keepdim=True) * nrm
        o[idx] = x + dd * 1e-4; d[idx] = dd
        alive[idx[lacc[idx] > Lmax]] = False
    mfp = float(torch.cat(seglen).mean()) if seglen else float('nan')
    st = dict(mfp=mfp, a=torch.cat(hits_a).mean(0).cpu().numpy(), s=torch.cat(hits_s).mean(0).cpu().numpy(),
              escape=escaped / n_rays)
    chains = []
    for n in range(1, max_order + 1):
        if not cand[n - 1]:
            chains.append(torch.zeros(0, n, dtype=torch.long, device=dev)); continue
        u = torch.unique(torch.cat(cand[n - 1]), dim=0)
        gq = room.pgid[u]
        u, gq = (u[(gq[:, 1:] != gq[:, :-1]).all(1)], gq[(gq[:, 1:] != gq[:, :-1]).all(1)]) if n > 1 else (u, gq)
        _, inv = torch.unique(gq, dim=0, return_inverse=True)   # one chain per reflector-group sequence
        first = torch.full((int(inv.max()) + 1 if len(inv) else 0,), len(inv), dtype=torch.long, device=dev)
        first.scatter_reduce_(0, inv, torch.arange(len(inv), device=dev), 'amin')
        chains.append(u[first])
    return chains, st


@torch.no_grad()
def diffuse_rain(room, src, rpts, n_rays=20000, max_depth=200, t_max=0.105, bin_s=1e-3, r_rec=0.3, seed=0,
                 thr=1e-9, scat_override=None, by_depth=False, moments=False):
    """pygsound-style diffuse ray tracing (max_depth reflections). Each ray carries per-band energy split into a
    pure-specular part Ep (reproduced exactly by the image sources, never collected here) and a scattered part Ed.
    At every hit: absorption (1-a); the Lambertian share s (Ep+Ed)(1-a) is sent to every visible receiver point by
    'diffuse rain' (next-event estimation, energy cos(theta) / (pi d^2) per unit area); the ray continues specularly
    with prob 1-s_mid or Lambertian with prob s_mid (energies re-weighted per band), and the scattered energy Ed that
    continues specularly is collected by receiver spheres (radius r_rec). Units: squared pressure amplitude of the
    1/(4 pi d) convention (direct sound = 1/(4 pi d)^2). Returns energy histogram H [R, 8, nbins] (bin_s bins, time from
    emission), and the number of reflections actually traced; with moments=True also the energy-weighted arrival
    direction D [R, 8, nbins, 3] (sum of energy x unit vector from the receiver towards the arrival)."""
    dev = room.dev
    rpts = torch.as_tensor(rpts, dtype=torch.float64, device=dev); R = rpts.shape[0]
    nb = int(math.ceil(t_max / bin_s))
    H = torch.zeros((max_depth + 1 if by_depth else 1) * R * 8 * nb, dtype=torch.float64, device=dev)
    Dm = torch.zeros(R * 8 * nb * 3 if moments else 0, dtype=torch.float64, device=dev)
    air = torch.as_tensor(AIR_DB_M, dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    d = torch.randn(n_rays, 3, device=dev, generator=g, dtype=torch.float64); d = d / d.norm(dim=-1, keepdim=True)
    o = torch.as_tensor(src, dtype=torch.float64, device=dev).expand(n_rays, 3).clone()
    lacc = torch.zeros(n_rays, dtype=torch.float64, device=dev)
    Ep = torch.full((n_rays, 8), 1.0 / n_rays, dtype=torch.float64, device=dev)
    Ed = torch.zeros_like(Ep)
    spec_leg = torch.zeros(n_rays, dtype=torch.bool, device=dev)  # current leg left its last surface specularly
    Lmax = C * t_max
    fa, fs = room.face_a, (room.face_s if scat_override is None else torch.full_like(room.face_s, scat_override))
    ridx = torch.arange(R, device=dev); bidx = torch.arange(8, device=dev)

    def deposit(rr, ee, tt, k, uu=None):  # rr [m] receiver, ee [m,8] energy, tt [m] time (s), k = number of reflections
        tb = (tt / bin_s).long()
        m = tb < nb
        rr, ee, tb = rr[m], ee[m], tb[m]
        key = (rr[:, None] * 8 + bidx[None]) * nb + tb[:, None]
        H.index_add_(0, (key + (k * R * 8 * nb if by_depth else 0)).reshape(-1), ee.reshape(-1))
        if moments:  # uu [m,3]: unit vector from the receiver towards the arrival
            k3 = (key[..., None] * 3 + torch.arange(3, device=dev)).reshape(-1)
            Dm.index_add_(0, k3, (ee[..., None] * uu[m][:, None, :]).reshape(-1))

    depth = 0
    for k in range(max_depth + 1):
        idx = (lacc < Lmax).nonzero().squeeze(1)
        if len(idx) == 0:
            break
        t, f = room.nearest_hit(o[idx], d[idx])
        ok = f >= 0
        t = torch.where(ok, t.double(), torch.full_like(t.double(), 1e3))
        # receiver spheres for scattered energy on specular legs
        sl = spec_leg[idx] & (Ed[idx].sum(1) > 0)
        if sl.any():
            ii = idx[sl]
            dist, u = _seg_point_dist(o[ii], d[ii], t[sl], rpts)
            hit = dist < r_rec
            if hit.any():
                a_, r_ = hit.nonzero(as_tuple=True)
                L = lacc[ii[a_]] + u[a_, r_]
                e = Ed[ii[a_]] / (math.pi * r_rec ** 2) / (4 * math.pi) * 10 ** (-air[None] * L[:, None] / 10)
                deposit(r_, e, L / C, k, -d[ii[a_]])
        idx, t, f = idx[ok], t[ok], f[ok]
        lacc[(lacc < Lmax).nonzero().squeeze(1)[~ok]] = 1e9  # escaped
        if k == max_depth or len(idx) == 0:
            break
        depth = k + 1
        x = o[idx] + d[idx] * t[:, None]
        lacc[idx] += t
        nrm = room.pn[room.fplane[f]]
        cin = (d[idx] * nrm).sum(-1, keepdim=True)
        nrm = nrm * -torch.sign(cin)                      # normal towards the incoming side
        a = fa[f]; s = fs[f]
        Etot = (Ep[idx] + Ed[idx]) * (1 - a)
        # diffuse rain to all receivers
        n_ = len(idx)
        xo = x + nrm * 1e-4
        P = rpts[None].expand(n_, R, 3).reshape(-1, 3); X = xo[:, None].expand(n_, R, 3).reshape(-1, 3)
        v = P - X; dist = v.norm(dim=-1)
        cos = ((v / dist[:, None]) * nrm.repeat_interleave(R, 0)).sum(-1)
        cand = (cos > 0) & (lacc[idx].repeat_interleave(R) + dist < Lmax)
        ci = cand.nonzero().squeeze(1)
        if len(ci):
            vis = ~room.segment_blocked(X[ci], P[ci], eps=1e-4)
            ci = ci[vis]
            ri, rr = ci // R, ci % R
            L = lacc[idx[ri]] + dist[ci]
            e = Etot[ri] * s[ri] * (cos[ci] / (math.pi * dist[ci] ** 2) / (4 * math.pi))[:, None] \
                * 10 ** (-air[None] * L[:, None] / 10)
            deposit(rr, e, L / C, k + 1, -(v[ci] / dist[ci, None]))
        # continue: specular with prob 1-s_mid, Lambertian with prob s_mid
        smid = s[:, 3:6].mean(1).clamp(0.02, 0.98)
        lam = torch.rand(n_, device=dev, dtype=torch.float64, generator=g) < smid
        w_sp = (1 - s) / (1 - smid)[:, None]; w_la = s / smid[:, None]
        Ep_new = torch.where(lam[:, None], torch.zeros_like(Etot), Ep[idx] * (1 - a) * w_sp)
        Ed_new = torch.where(lam[:, None], Etot * w_la, Ed[idx] * (1 - a) * w_sp)
        Ep[idx] = Ep_new; Ed[idx] = Ed_new
        dsp = d[idx] - 2 * (d[idx] * room.pn[room.fplane[f]]).sum(-1, keepdim=True) * room.pn[room.fplane[f]]
        r1 = torch.rand(n_, device=dev, dtype=torch.float64, generator=g); r2 = torch.rand(n_, device=dev, dtype=torch.float64, generator=g)
        tmp = torch.where(nrm[:, :1].abs() < 0.9, torch.tensor([[1., 0, 0]], device=dev, dtype=torch.float64).expand(n_, 3),
                          torch.tensor([[0., 1, 0]], device=dev, dtype=torch.float64).expand(n_, 3))
        t1 = torch.cross(nrm, tmp, dim=-1); t1 = t1 / t1.norm(dim=-1, keepdim=True); t2 = torch.cross(nrm, t1, dim=-1)
        ph = 2 * math.pi * r1; sr = torch.sqrt(r2)
        dla = t1 * (sr * torch.cos(ph))[:, None] + t2 * (sr * torch.sin(ph))[:, None] + nrm * torch.sqrt(1 - r2)[:, None]
        dn = torch.where(lam[:, None], dla, dsp)
        d[idx] = dn; o[idx] = x + dn * 1e-4
        spec_leg[idx] = ~lam
        dead = (Ep[idx] + Ed[idx]).max(1).values * n_rays < thr
        lacc[idx[dead]] = 1e9
    Hout = H.view(max_depth + 1, R, 8, nb) if by_depth else H.view(R, 8, nb)
    return (Hout, depth, Dm.view(R, 8, nb, 3)) if moments else (Hout, depth)
