### Models and ensembles (test set)

| model | kappa | accuracy | macro F1 | referable AUC | train time | seeds |
|---|---|---|---|---|---|---|
| ConvNeXt-T | 0.803 +/- 0.022 | 0.676 +/- 0.006 | 0.569 +/- 0.018 | 0.957 +/- 0.009 | 201s | 3 |
| EfficientNetV2-S | 0.777 +/- 0.032 | 0.653 +/- 0.017 | 0.520 +/- 0.029 | 0.949 +/- 0.007 | 204s | 3 |
| MaxViT-T (hybrid) | 0.816 +/- 0.013 | 0.681 +/- 0.026 | 0.562 +/- 0.015 | 0.968 +/- 0.005 | 364s | 3 |
| ResNet50 | 0.755 +/- 0.032 | 0.615 +/- 0.035 | 0.479 +/- 0.048 | 0.925 +/- 0.012 | 146s | 3 |
| ensemble: all four | 0.809 +/- 0.014 | 0.686 +/- 0.013 | 0.566 +/- 0.024 | 0.967 +/- 0.002 | - | 3 |
| ensemble: ConvNeXt-T + MaxViT-T (CNN + hybrid) | 0.812 +/- 0.023 | 0.691 +/- 0.005 | 0.584 +/- 0.013 | 0.970 +/- 0.003 | - | 3 |
| ensemble: ResNet50 + EfficientNetV2-S (CNNs) | 0.780 +/- 0.030 | 0.662 +/- 0.011 | 0.534 +/- 0.028 | 0.952 +/- 0.002 | - | 3 |

### Head: class logits vs ordinal regression

| variant | kappa | accuracy | macro F1 | referable AUC | seeds |
|---|---|---|---|---|---|
| class_logits | 0.816 +/- 0.013 | 0.681 +/- 0.026 | 0.562 +/- 0.015 | 0.968 +/- 0.005 | 3 |
| ordinal | 0.834 +/- 0.019 | 0.652 +/- 0.023 | 0.516 +/- 0.021 | 0.973 +/- 0.000 | 3 |

### Preprocessing

| variant | kappa | accuracy | macro F1 | referable AUC | seeds |
|---|---|---|---|---|---|
| ben | 0.827 +/- 0.020 | 0.702 +/- 0.013 | 0.564 +/- 0.036 | 0.971 +/- 0.002 | 3 |
| none | 0.816 +/- 0.013 | 0.681 +/- 0.026 | 0.562 +/- 0.015 | 0.968 +/- 0.005 | 3 |

### GAN recipes

| recipe | FID | FID at midpoint | nearest-train dist (generated) | nearest-train dist (unseen real) |
|---|---|---|---|---|
| A_original | 122.3 | 300: 171.5, 600: 121.9 | 7.53 | 6.73 |
| B_hinge_sn | 117.8 | 300: 145.3, 600: 117.6 | 8.12 | 6.73 |
| C_diffaug | 146.1 | 300: 129.3, 600: 146.3 | 10.77 | 6.73 |
| D_full | 147.6 | 300: 194.5, 600: 146.9 | 10.77 | 6.73 |

### Does GAN data help (64px, ResNet18)

| training data | kappa | accuracy | macro F1 | recall per grade 0-4 | seeds |
|---|---|---|---|---|---|
| real + GAN images | 0.604 +/- 0.036 | 0.538 +/- 0.020 | 0.394 +/- 0.026 | 0.73 / 0.19 / 0.57 / 0.35 / 0.18 | 5 |
| real only | 0.564 +/- 0.039 | 0.531 +/- 0.022 | 0.391 +/- 0.041 | 0.73 / 0.18 / 0.50 / 0.42 / 0.14 | 5 |
| real, class-balanced sampling | 0.598 +/- 0.028 | 0.496 +/- 0.016 | 0.401 +/- 0.018 | 0.67 / 0.46 / 0.37 / 0.31 / 0.30 | 5 |
