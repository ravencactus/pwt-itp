# pwt-itp: moving-microphone measurement of spatial room impulse responses

Code, simulator, dataset, and results for the paper

> R. Zheng, Z. Li, S. Zhao, and J. Wang, "Moving-Microphone Measurement of Spatial Room Impulse Responses:
> Identifiability and Transport Priors," submitted to *IEEE Transactions on Audio, Speech and Language Processing*.

A first-order Ambisonic (FOA, B-format) microphone moves along a known path while a known excitation is played, and
the time-varying spatial room impulse response (SRIR) is estimated from the recording. Between two knots of the
time-varying model, a plane wave arriving from direction **u** is delayed by the projection of the array
displacement **d** onto **u**. The estimators in this repository use this transport as the prior of a maximum a
posteriori (MAP) estimate:

* **PWT** (plane-wave transport prior): components on a grid of Q directions, each delayed exactly; used in the
  low band;
* **ITP** (intensity-guided transport prior): one direction per knot and lag, taken from the active intensity of the
  current estimate; used in the high band;
* **PWT+ITP**: zero-phase two-band combination of both.

Reference methods: random-walk MAP (RW), first-order diffuse-field LMMSE prior (FOA-LMMSE), NLMS, Kalman filter,
frequency-domain Kalman filter, and moving-microphone kernel ridge regression (KRR, KRR-D).

## Layout

```
pwtitp/                 estimators and simulator
  model.py              time-varying FIR model, conjugate gradients, random-walk MAP, active intensity
  pwt.py                plane-wave transport prior
  itp.py                intensity-guided transport prior
  pipeline.py           RW -> PWT -> ITP -> crossover; estimate() runs the complete estimator
  lmmse.py              first-order diffuse-field LMMSE prior (FOA-LMMSE)
  baselines/            NLMS, Kalman filter, frequency-domain Kalman filter, kernel ridge regression
  metrics.py            NMSE measures
  trajectorir.py        loading and evaluation protocol of the trajectoRIR database
  sim.py                moving FOA recordings: scenario(), add_noise(), speech excitation
  mesh/                 simulator of the furnished houses
    geometry.py         mesh image sources with exact visibility, diffuse-rain ray tracing, Eyring T60
    bank.py             source and path sampling, scene construction
    render.py           FOA rendering along a stored path
    scenario.py         moving recordings from the dataset; load_house()
examples/               trajectorir_demo.py: RW and PWT+ITP on one measured recording
experiments/            run_real.py, run_real_lmmse.py, real_*.py: measured recordings
  simulation/           runs on the simulated houses; build_bank.py builds the dataset from 3D-FRONT
  jobs.sh               every run behind the paper
tools/export_dataset.py conversion of build_bank.py output to the released .npz format (lossless)
results/
  raw/                  raw results of all runs in the paper (gzipped JSON lines, one record per grid point)
  example.npz           estimates of the example figure
paper/                  collect.py -> outputs/numbers.json; make_tables.py, make_figs.py -> outputs/
data/speech_manifest.json   LibriSpeech test-clean utterances used as speech excitation
```

## Requirements

Python 3.8+, PyTorch 2 with a CUDA GPU (24 GB suffice for Q = 128 and K = 1600), NumPy, SciPy, soundfile,
Matplotlib:

```
pip install -r requirements.txt
```

Building the dataset from 3D-FRONT additionally needs NVIDIA Warp (`pip install warp-lang`); rendering recordings
from the released dataset does not.

## Usage

```python
from pwtitp import Fwd, Solver, estimate

# x: excitation, length N + K - 1;  y: B-format recording (N, 4), SN3D, room frame;
# dp: displacement between consecutive knots ((J - 1), 3) in m, with N = (J - 1) H.  All torch tensors on the GPU.
F = Fwd(x, K, H, J)
S = Solver(F, y)
h, h_pwt, h_itp = estimate(S, F, dp, fs, c=343., fc=1000.)    # (J, K, 4) B-format knot responses
```

Regularization weights are given per knot spacing H; the values selected in the paper are recorded in
`outputs/numbers.json` after running `paper/collect.py`.

Measured example (trajectoRIR database, Damiano et al., 2026; set `TRAJECTORIR` to the database root that contains
`audio/` and `meta/`):

```
python examples/trajectorir_demo.py V1
# V1: NMSE 150 Hz - 2 kHz   RW -11.4 dB   PWT+ITP -11.7 dB
```

## Dataset of furnished houses

600 furnished houses from 3D-FRONT with the acoustic materials of GWA. Each house has two sources and about six
trajectories of 3.4 m (four at least 0.7 m from their source, and where possible two that pass it at 0.25-0.5 m).
For every trajectory the dataset stores the image sources up to order 50 with their octave-band gains and their
exact visibility on a 1-mm grid along the path, the diffuse-rain energy envelopes and arrival directions, candidate
point scatterers with their visibility, and the band reverberation times of the house. Moving FOA recordings and
their true time-varying responses are rendered from it at any speed, excitation, and noise level.

Download: Zenodo, DOI ZENODO_DOI (about ZENODO_SIZE GB). Unpack it and set `PWTITP_HOUSES` to the folder:

```
houses/<house id>.npz    one file per house; key 's<i>/<field>' for scene (trajectory) i
index.txt                house order used in the paper (houses 1-4: development, 5-40: test, 41-600: unused)
houses.jsonl             per-house statistics (size, faces, band T60, number of trajectories)
```

```python
import os
os.environ['PWTITP_HOUSES'] = '/path/to/houses'
from pwtitp.sim import scenario, add_noise
s = scenario(507, 0.8, 'wn', K=1600)        # house 5 of index.txt, 0.8 m/s, white noise
y = add_noise(s['y'], 20., 5)               # 20 dB SNR; s['ht']: true knot responses (J, K, 4)
```

By default the first 40 houses of `index.txt` are loaded, as in the paper (`PWTITP_N_HOUSES` changes this).
To rebuild the dataset from 3D-FRONT (`MESH_ROOT/<id>/house_sp.obj` with GWA materials):
`python experiments/simulation/build_bank.py MESH_ROOT OUT 600`, then `python tools/export_dataset.py`.

## Reproducing the paper

Tables and figures from the raw results:

```
cd paper
python collect.py         # dev/test selection of every hyperparameter -> ../outputs/numbers.json
python make_tables.py     # -> ../outputs/tables/
python make_figs.py       # -> ../outputs/figs/
```

Hyperparameters of every method are selected on the development houses 1-4 and reported on the test houses
(5-8 for all conditions, 5-40 at 20 dB SNR). For the measured recordings they are selected on loudspeaker SR and
reported on loudspeaker SL. In the raw results, `RW-MAP`, `LMMSE1`, `KRR-M`, and `KRR-MD` denote RW, FOA-LMMSE, KRR,
and KRR-D.

Re-running the experiments: `experiments/jobs.sh` lists every run. The measured-data runs need the trajectoRIR
database, the simulated runs need the dataset (`PWTITP_HOUSES`) and LibriSpeech test-clean (`LIBRISPEECH`, the folder
that contains `test-clean/`; the utterances are listed in `data/speech_manifest.json`).

## License

Code: MIT (see `LICENSE`). Dataset: CC BY-NC 4.0; it is derived from 3D-FRONT and GWA, whose terms of use also apply.

## Citation

```bibtex
@article{zheng_moving_srir,
  author  = {Zheng, Ruiqi and Li, Zhiyu and Zhao, Shenghui and Wang, Jing},
  title   = {Moving-Microphone Measurement of Spatial Room Impulse Responses: Identifiability and Transport Priors},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  note    = {submitted}
}
```
