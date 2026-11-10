"""Scene bank of furnished 3D-FRONT houses for the simulations (one .pt per accepted house, one json line per house).
House filter: all used materials carry 8-band sound_a / sound_s; height 2.2-4.5 m; floor area 8-400 m^2; closed mesh
around every source (< 1 % of its rays escape); broadband Eyring T60 (500 Hz-1 kHz) in 0.15-2.0 s.
Houses are visited in the order of md5(house id). Per house: 2 sources, each with 2 paths at >= 0.7 m from the source and
(if possible) 1 path that passes 0.25-0.5 m from it, all 3.4 m long.
usage: build_bank.py MESH_ROOT OUT n_houses [shard nshard]    (MESH_ROOT/<id>/house_sp.obj)"""
import _common
import sys, os, json, time, hashlib, traceback
import numpy as np, torch
from pwtitp.mesh.geometry import MeshRoom
from pwtitp.mesh.bank import build_scenes

D, OUT, n_want = sys.argv[1], sys.argv[2], int(sys.argv[3])
shard, nshard = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
os.makedirs(OUT, exist_ok=True)
h = lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16)
ids = sorted(os.listdir(D), key=h)[shard::nshard]
log_f = f'{OUT}/log_{shard}.jsonl'
done = {json.loads(l)['id'] for l in open(log_f)} if os.path.exists(log_f) else set()
n_ok = len([f for f in os.listdir(OUT) if f.endswith('.pt')])
log = open(log_f, 'a')
for rid in ids:
    if n_ok >= n_want: break
    if rid in done: continue
    t0 = time.time(); rec = dict(id=rid, ok=False)
    try:
        room = MeshRoom(f'{D}/{rid}/house_sp.obj', device='cuda')
        sz = room.bounds[1] - room.bounds[0]
        rec.update(faces=int(room.v0.shape[0]), size=sz.round(2).tolist())
        if not room.colored: rec['why'] = 'materials'; raise StopIteration
        if not (2.2 <= sz[2] <= 4.5 and 8 <= sz[0]*sz[1] <= 400): rec['why'] = 'size'; raise StopIteration
        rng = np.random.default_rng(h(rid) % (1 << 32))
        for att in range(30):   # shared GPU: wait and retry on OOM (same rng seed -> same scenes)
            try:
                rng = np.random.default_rng(h(rid) % (1 << 32))
                sc, st = build_scenes(room, rng, n_src=2, n_far=2, n_near=1, L=3.4); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); time.sleep(60)
        else: raise RuntimeError('oom')
        if st is None or len(sc) < 4: rec['why'] = 'no_paths'; raise StopIteration
        t60 = room.eyring_t60(st); rec.update(escape=st['escape'], mfp=st['mfp'], t60=t60.round(3).tolist(), n=len(sc))
        if not 0.15 <= float(np.mean(t60[3:5])) <= 2.0: rec['why'] = 't60'; raise StopIteration
        torch.save(sc, f'{OUT}/{rid}.pt'); rec['ok'] = True; n_ok += 1
    except StopIteration: pass
    except Exception as e:
        if 'out of memory' in repr(e) or 'oom' in repr(e):  # not a property of the house: retry on the next run
            print('skip (oom)', rid, flush=True); torch.cuda.empty_cache(); time.sleep(30); continue
        rec['why'] = 'error: ' + repr(e)[:200]; traceback.print_exc()
    rec['sec'] = round(time.time() - t0, 1)
    log.write(json.dumps(rec) + '\n'); log.flush(); print(rec, flush=True)
    torch.cuda.empty_cache()
print('done', n_ok, flush=True)
