# STEAD example waveforms

The HDF5 contains 14 real three-component traces, each 6000 samples at 100 Hz. `sample.csv` contains the full sample list and original split membership. `train.csv`, `validation.csv` and `test.csv` index the same HDF5 by `trace_name`.

| Split | Earthquake | Noise |
| --- | ---: | ---: |
| Train | 8 | 1 |
| Validation | 2 | 0 |
| Test | 2 | 1 |

Selection used seed 42. Each earthquake has valid P and S and a distinct `source_id`. Each noise trace has neither pick. The two manifests record source split and row, HDF5 keys, waveform hashes, and output hashes. [Data licence and attribution](DATA_LICENSE.md).
