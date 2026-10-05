# Introduction

This model is trained using the dataset from
https://en.data-baker.com/datasets/freeDatasets/

The dataset contains 10000 Chinese sentences of a native Chinese female speaker,
which is about 12 hours.

**Note**: The dataset is for non-commercial use only.

You can find the training code at
https://github.com/k2-fsa/icefall/tree/master/egs/baker_zh/TTS

## Local deployment (2026-10-02)

Runtime: sherpa-onnx 1.13.8. The vocoder is vocos-22khz-univ.onnx from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models .

Four lexicon entries used the missing token `shei2`; these were changed to the
supported alternative pronunciation `shui2` to avoid initialization warnings.
The neural network weights are unchanged.
