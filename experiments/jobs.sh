#!/bin/bash
# Every run behind the paper. Each line is independent; run them in parallel as resources allow.
# Measured recordings (trajectoRIR; env TRAJECTORIR):
cd "$(dirname "$0")"; O=../results/raw; mkdir -p $O/real $O/lmmse
for V in V1 V2 V3; do
  python run_real.py $V $O/real/real_$V.jsonl
  python run_real_lmmse.py $V $O/lmmse/real_${V}_lmmse.jsonl
done
python real_diagnostics.py $O/real/real_diag.jsonl
python real_egonoise.py > $O/real/real_egonoise.txt

# Simulated houses: need the dataset (env PWTITP_HOUSES) and LibriSpeech test-clean (env LIBRISPEECH).
cd simulation; mkdir -p ../$O/sim ../$O/sim_ext ../$O/sim_ext6
S=../$O/sim
for kind in wn speech; do for v in 0.2 0.4 0.8; do
  python run_baselines.py $v $kind $S/base_main_${kind}_v$v.jsonl 1,2,3,4,5,6,7,8
  python run_main.py      $v $kind $S/main_${kind}_v$v.jsonl 1,2,3,4,5,6,7,8
  NOPW=1 python run_main.py $v $kind $S/mainb_${kind}_v$v.jsonl 1,2,3,4,5,6,7,8
  python run_ablation.py  $v $kind $S/abl_${kind}_v$v.jsonl
  python run_ext.py  $v $kind ../$O/sim_ext/ext_${kind}_v$v.jsonl 1,2,3,4,5,6,7,8
  python run_ext6.py $v $kind ../$O/sim_ext6/ext6_${kind}_v$v.jsonl 1,2,3,4,5,6,7,8
  python run_lmmse.py $v $kind ../$O/lmmse/lm_main_${kind}_v$v.jsonl 1,2,3,4,5,6,7,8
  SNRS=20 python run_baselines.py $v $kind ../$O/sim_ext/b20_${kind}_v$v.jsonl $(seq -s, 9 40)
  SNRS=20 python run_main.py      $v $kind ../$O/sim_ext/m20_${kind}_v$v.jsonl $(seq -s, 9 40)
  SNRS=20 NOPW=1 python run_main.py $v $kind ../$O/sim_ext/mb20_${kind}_v$v.jsonl $(seq -s, 9 40)
  SNRS=20 python run_ext.py  $v $kind ../$O/sim_ext/x20_${kind}_v$v.jsonl $(seq -s, 9 40)
  SNRS=20 python run_ext6.py $v $kind ../$O/sim_ext6/x620_${kind}_v$v.jsonl $(seq -s, 9 40)
  SNRS=20 python run_lmmse.py $v $kind ../$O/lmmse/lm20_${kind}_v$v.jsonl $(seq -s, 9 40)
done; done
for v in 0.1 0.6 1.2 1.6; do
  SNRS=20 python run_baselines.py $v wn $S/base_sweep_wn_v$v.jsonl 1,2,3,4,5,6,7,8
  SNRS=20 python run_main.py      $v wn $S/sweep_wn_v$v.jsonl 1,2,3,4,5,6,7,8
  SNRS=20 NOPW=1 python run_main.py $v wn $S/sweepb_wn_v$v.jsonl 1,2,3,4,5,6,7,8
  SNRS=20 python run_ext.py  $v wn ../$O/sim_ext/extsweep_wn_v$v.jsonl 1,2,3,4,5,6,7,8
  SNRS=20 python run_ext6.py $v wn ../$O/sim_ext6/ext6sweep_wn_v$v.jsonl 1,2,3,4,5,6,7,8
  SNRS=20 python run_lmmse.py $v wn ../$O/lmmse/lm_sweep_wn_v$v.jsonl 1,2,3,4,5,6,7,8
done
for kind in wn speech; do for v in 0.4 0.8; do for geo in near far; do
  python run_near.py $v $kind $geo $S/near_${geo}_${kind}_v$v.jsonl
done; done; done
python example.py ../../results/example.npz
