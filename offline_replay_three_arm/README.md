# offline_replay_three_arm/

Support files for `offline_replay_three_arm.ipynb` (the folder must sit next to the notebook, or set `SUPPORT_DIR`).
The notebook reads them; nothing here is copied into `prime/`.

| file | what it is |
|---|---|
| `post_5khz.patch` | `git diff 11bf662 ee41369` of the prime repository: the post-stimulus window at 5 kHz (branch `post-5khz`). Applies to `11bf662` and to `f4fd395` (they differ only in `config.py`). |
| `post_lowpass.patch` | the low-pass integration patch (applies on top of `ee41369`): settings in `prime.yaml` / `prime_config.py`, design + freeze in `Preprocessor.calibrate()`, filter after the first blank in `preprocess_calibration()` and `preprocess_post()`, stamp check, `from_bundle()` restore, `proc_info` entry. |
| `post_lowpass.py` | the filter module (`PostLowpass`), served to the notebook as `prime_core.preprocessing.utils.post_lowpass`. |

The notebook applies the hunks of both patches to `prime_config.py`, `resampling.py`, `preprocessor.py` and
`dipole_fitter.py` in memory, at import time, and skips a patch the prime code already contains. The hunks for
`prime.yaml` and `2_intervention.py` are not used there; they belong to the real patch for the decider.

Replace these files only together with the notebook: a patch that does not fit the prime code stops the notebook
at the first hunk that does not match.
