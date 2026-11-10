"""Export the scene bank (torch .pt, one per house) to the released format: one compressed .npz per house.
Each scene s of a house is stored with keys 's{s}/{field}'; scalars as 0-d arrays, strings as unicode arrays.
usage: export_dataset.py MANIFEST BANK_DIR [BANK_DIR ...] OUT [shard nshard]"""
import sys, os, json, numpy as np, torch
torch.set_num_threads(2)

def to_npz(scenes, path):
    arr = {}
    for i, s in enumerate(scenes):
        for k, v in s.items():
            arr[f's{i}/{k}'] = v.numpy() if torch.is_tensor(v) else np.asarray(v)
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arr); os.replace(tmp, path)

def check(scenes, path):
    z = np.load(path)
    for i, s in enumerate(scenes):
        for k, v in s.items():
            a = z[f's{i}/{k}']
            if torch.is_tensor(v):
                assert a.dtype == v.numpy().dtype and np.array_equal(a, v.numpy()), (path, i, k)
            else:
                assert a.item() == v, (path, i, k)

if __name__ == '__main__':
    args = sys.argv[1:]
    shard, nshard = 0, 1
    if len(args) > 3 and args[-1].isdigit() and args[-2].isdigit():
        shard, nshard = int(args[-2]), int(args[-1]); args = args[:-2]
    man, banks, out = args[0], args[1:-1], args[-1]
    os.makedirs(out, exist_ok=True)
    ids = sorted(open(man).read().split())[shard::nshard]
    for rid in ids:
        dst = f'{out}/{rid}.npz'
        if os.path.exists(dst): continue
        src = next(f'{b}/{rid}.pt' for b in banks if os.path.exists(f'{b}/{rid}.pt'))
        sc = torch.load(src, weights_only=False)
        to_npz(sc, dst); check(sc, dst)
        print(rid, os.path.getsize(src) >> 20, '->', os.path.getsize(dst) >> 20, 'MB', flush=True)
    print('done', flush=True)
