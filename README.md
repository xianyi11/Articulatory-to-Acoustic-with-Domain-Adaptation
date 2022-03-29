# B-mode Ultrasound Tongue Imaging-Based Towards Speaker-Independent Articulatory-to-Acoustic Mapping Using Domain Adaptation

## Introduction
The articulatory-to-acoustic mapping refers to estimating the speech signals leveraging the articulatory information, and its applications are evident in several fields, such as silent speech recognition and synthesis. In this paper, we aim to predict the mel-spectrogram of the audio signals, using midsagittal B-mode ultrasound tongue images of the vocal tract as the input. Recently, deep learning has become the choice of methodology for the estimation task. However, most previous attempts have been constrained to the speaker-dependent scenario, and the performance is greatly decreased for unseen speakers. Here, we present a novel approach towards being speaker-independent, using domain-adaptation and adversarial learning. Objective evaluation is conducted to demonstrate the effectiveness of the proposed method, and three evaluation metrics are used, including the MSE, Structural Similarity Index (SSIM), and complex wavelet Structural Similarity Index (CW-SSIM). The results indicate that our proposed method can achieve superior performance under the speaker-independent scenario.


## Model Architecture
![image](https://user-images.githubusercontent.com/74498528/160514541-c93b8591-c545-4f8d-a246-6dde6a464760.png)

## result

Comparison results in term of mean MSE (Lower is better), SSIM and CW-SSIM (Both metrics ranges between 0 and 1. Higher value denotes better performance, while 1 represents the predicted one is the same as the ground-truth)

![image](https://user-images.githubusercontent.com/74498528/160514706-4684b595-f7eb-4727-a683-828bfa760615.png)


compare with the ground-truth
![image](https://user-images.githubusercontent.com/74498528/160410707-e5af1791-2bd6-4be7-a858-7d1697b16a55.png)
