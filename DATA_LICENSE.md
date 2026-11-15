# Data licensing

The redistributed example under [data/sample_stead](data/sample_stead/) is derived from the Stanford Earthquake Dataset (STEAD) by S. Mostafa Mousavi, Yixiao Sheng, Weiqiang Zhu and Gregory C. Beroza. Cite *STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI*, IEEE Access (2019). [Dataset and upstream licence](https://github.com/smousavi05/STEAD/blob/master/LICENSE) · [Dataset article](https://doi.org/10.1109/ACCESS.2019.2947848).

The sample files are offered under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Waveforms were selected from an existing STEAD split and repackaged in one losslessly compressed HDF5 file. The subset retains trace identity, category, event identity when present, and arrival picks. Arrival fields are represented as integer sample indices, with missing noise arrivals encoded as `-1`. Separate split CSV files and extraction manifests were added.

The project software and EQTransformer code have separate licences. Shanxi data and formal Shanxi caches are external restricted inputs. Contact their rights holders for access and any authorisation to redistribute data or derived artifacts.
