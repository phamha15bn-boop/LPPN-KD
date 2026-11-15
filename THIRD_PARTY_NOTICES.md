# Third-party notices

## LPPN / csnbench

The Student model and supervised loss derive from the LPPNM implementation in `cangyeone/csnbench`. The pinned source is `models/LPPNM.py` at commit `66614c5c1ffc13fcabd0f4c0b7a599e17bebb839` (2022-07-28). The archived precursor's SHA-256 is `8475d1c5e5d2ea1e9d57b119e55be2985d6e80f96a6693b2f6fcf6bd6dbea8b8`. The LPPN author repository credits Ziye Yu and Weitao Wang at CEA-IGP. The pinned csnbench model file carries no named copyright header. The root [GPL-3.0 licence](LICENSE) is retained for the derived implementation. [Pinned model](https://github.com/cangyeone/csnbench/blob/66614c5c1ffc13fcabd0f4c0b7a599e17bebb839/models/LPPNM.py) · [Pinned GPL text](https://github.com/cangyeone/csnbench/blob/66614c5c1ffc13fcabd0f4c0b7a599e17bebb839/LICENSE) · [LPPN author attribution](https://github.com/cangyeone/LPPN#authors)

Qineng Song adapted the Student wrapper and forward interfaces in `lppn/model.py` and separated and adapted the supervised loss in `lppn/loss.py`. The documented modification period is **2026**. The regression-dimension and logits changes were recorded in the 2026-09-25 preflight. The release copy of the model and loss was documented on 2026-11-15. Exact dates for individual changes are incomplete. The original LPPN, csnbench source and their original attribution remain with their upstream authors.

## EQTransformer

The vendored subset under `teacher/eqtransformer/EQTransformer/` comes from the local EQTransformer adaptation of S. Mostafa Mousavi's project. Its [independent MIT licence](teacher/eqtransformer/LICENSE) retains the upstream copyright line, `Copyright (c) 2020-2025 S.Mostafa Mousavi`. The local source snapshot is established. Its exact upstream Git commit and software release cannot be assigned uniquely. The initializer's `0.0.1` field records a package value. [Upstream repository](https://github.com/smousavi05/EQTransformer) · [Upstream MIT text](https://github.com/smousavi05/EQTransformer/blob/2d1fa094a705798e68757e6b3c802361810cf8c3/LICENSE) · [EQTransformer paper](https://doi.org/10.1038/s41467-020-17591-w)

Qineng Song's local adaptations cover fixed split reading, HDF5 waveform handling, tail batches, Noise and missing arrival-label handling, the teacher inference and feature interfaces, and their integration into the release workflow. The original model, training utilities and embedded third-party implementations retain their respective attribution and licences.

## Keras helper components

EQTransformer already included these adapted implementations. Qineng Song integrated the vendored EQTransformer teacher with the LPPN-KD workflow. The Git references below identify comparable historical source files and the corresponding licence texts. The exact embedded class revisions remain unconfirmed.

### LayerNormalization

The adapted class in `EqT_utils.py` comes from `CyberZHG/keras-layer-normalization`, and the applicable notice names **Zhao HG**. The EQTransformer implementation changes constructors, build and configuration options. [Historical source](https://github.com/CyberZHG/keras-layer-normalization/blob/133d1ce3a09574ce2bb2e741bb8e6a54282e8373/keras_layer_normalization/layer_normalization.py) · [MIT licence](https://github.com/CyberZHG/keras-layer-normalization/blob/133d1ce3a09574ce2bb2e741bb8e6a54282e8373/LICENSE)

```text
MIT License

Copyright (c) 2018 Zhao HG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### FeedForward

The adapted class in `EqT_utils.py` comes from `CyberZHG/keras-position-wise-feed-forward`, and the applicable notice names **Zhao HG**. The EQTransformer implementation changes constructors, build and configuration options. [Historical source](https://github.com/CyberZHG/keras-position-wise-feed-forward/blob/bb54de057ed16d103c144fb7763a18273b376076/keras_position_wise_feed_forward/feed_forward.py) · [MIT licence](https://github.com/CyberZHG/keras-position-wise-feed-forward/blob/bb54de057ed16d103c144fb7763a18273b376076/LICENSE)

```text
MIT License

Copyright (c) 2018 Zhao HG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### SeqSelfAttention

The adapted class in `EqT_utils.py` comes from `CyberZHG/keras-self-attention`, and its historical licence identifies **PoW** as the copyright holder. The EQTransformer implementation adapts initialization and additive-attention input-length handling. [Historical source](https://github.com/CyberZHG/keras-self-attention/blob/c66034d2554ff63033caf614a5dea2e57c5612dd/keras_self_attention/seq_self_attention.py) · [MIT licence](https://github.com/CyberZHG/keras-self-attention/blob/c66034d2554ff63033caf614a5dea2e57c5612dd/LICENSE)

```text
MIT License

Copyright (c) 2018 PoW

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BMC peak detector

The `_detect_peaks` function in `EqT_utils.py` embeds the BMC 1.0.4 peak detection calculation by **Marcos Duarte**, inherited through EQTransformer. The calculation matches `functions/detect_peaks.py` at historical commit `94e88df1ea62d89dfb4ec68c748bfe70ceba172a` apart from the function name and omission of plotting. Qineng Song integrated the vendored teacher. The original peak detector retains its own copyright and MIT licence. [Pinned BMC source](https://github.com/demotu/BMC/blob/94e88df1ea62d89dfb4ec68c748bfe70ceba172a/functions/detect_peaks.py) · [Historical MIT notice](https://github.com/demotu/BMC/blob/94e88df1ea62d89dfb4ec68c748bfe70ceba172a/LICENSE.txt)

```text
Copyright (c) 2013 Marcos Duarte

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
```

## STEAD dataset

The bundled demonstration repackages **14 STEAD traces** under **CC BY 4.0**, preserving waveform arrays and recording the selection, split and integer arrival-label representation. Credit S. Mostafa Mousavi, Yixiao Sheng, Weiqiang Zhu and Gregory C. Beroza and cite the STEAD paper. Shanxi waveforms and formal weights are excluded. [Local data licence and modifications](DATA_LICENSE.md) · [Official dataset and citation](https://github.com/smousavi05/STEAD) · [Official CC BY 4.0 notice](https://github.com/smousavi05/STEAD/blob/master/LICENSE) · [STEAD paper](https://doi.org/10.1109/ACCESS.2019.2947848)

## Original LPPN-KD contributions

`Copyright (c) 2026 Qineng Song` applies to the confirmed project additions. Qineng Song is the sole software author and maintainer of LPPN-KD. The original contributions include the local EQTransformer and STEAD data adaptations, teacher inference and feature extraction interfaces, probability and feature caches, KD-C, KD-FP and KD-CFP, Student training, prediction and evaluation adaptations, the STEAD demo, release tooling and reproducibility documentation. The upstream components above retain their own copyrights and licence notices.
