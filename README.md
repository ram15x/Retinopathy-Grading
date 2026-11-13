# Retinopathy Lab

Diabetic retinopathy grading from fundus photos, plus a small study of conditional GANs for making
synthetic retina images. This started as a single Colab notebook (kept in `legacy/`) that trained one
DCGAN in TensorFlow. I rebuilt it as a PyTorch project where every claim comes from a run you can repeat.

The old notebook used a private image folder that I no longer have, so everything here uses the public
RetinaMNIST dataset instead. Numbers are not comparable with the old notebook.

## What is in here

| experiment | question |
|---|---|
| E1 backbones | ResNet50, EfficientNetV2-S, ConvNeXt-T and MaxViT-T (a convolution and attention hybrid). Which one grades best? |
| E2 ensembles | Does averaging the models help? CNN only, CNN plus hybrid, all four. |
| E3 head | Plain 5-class logits or a single ordinal regression output? |
| E4 preprocessing | Does Ben Graham contrast normalisation help? |
| E5 GAN recipes | Four recipes for a conditional GAN, compared by FID. |
| E6 synthetic data | Does adding GAN images to the training set help a classifier, compared with just resampling? |

## Data

RetinaMNIST from MedMNIST v2, the 224 by 224 version. Five grades (no DR, mild, moderate, severe,
proliferative). The split is the one that ships with the dataset:

| split | no DR | mild | moderate | severe | proliferative | total |
|---|---|---|---|---|---|---|
| train | 486 | 128 | 206 | 194 | 66 | 1080 |
| val | 54 | 12 | 28 | 20 | 6 | 120 |
| test | 174 | 46 | 92 | 68 | 20 | 400 |

It is small and imbalanced. Only 20 test images are proliferative, so per-grade numbers are noisy.

## How the models are trained

All backbones start from ImageNet weights. AdamW, one-cycle learning rate, mixed precision, 25 epochs,
flips, small rotations and light colour jitter. The epoch is picked on the validation set, never on test.
For the ordinal head the cut points between grades are also fitted on validation. Ensembles average the
class probabilities. Each setting is run with 3 seeds and I report mean and standard deviation.

Metrics: quadratic weighted kappa (the headline, because grades are ordered and a 0 vs 4 mistake should
cost more than 0 vs 1), accuracy, macro F1, and AUC for referable DR (grade 2 and above).

## Results

Everything below is on the 400 test images, mean and standard deviation over seeds. The full tables are in
`results/RESULTS.md`.

### Models and ensembles

![backbones](docs/backbones.png)

| model | kappa | accuracy | macro F1 | referable AUC |
|---|---|---|---|---|
| ResNet50 | 0.755 +/- 0.032 | 0.615 | 0.479 | 0.925 |
| EfficientNetV2-S | 0.777 +/- 0.032 | 0.653 | 0.520 | 0.949 |
| ConvNeXt-T | 0.803 +/- 0.022 | 0.676 | 0.569 | 0.957 |
| MaxViT-T (hybrid) | 0.816 +/- 0.013 | 0.681 | 0.562 | 0.968 |
| ensemble, all four | 0.809 +/- 0.014 | 0.686 | 0.566 | 0.967 |
| ensemble, ConvNeXt-T + MaxViT-T | 0.812 +/- 0.023 | 0.691 | 0.584 | 0.970 |
| ensemble, ResNet50 + EfficientNetV2-S | 0.780 +/- 0.030 | 0.662 | 0.534 | 0.952 |

What I take from this:

- The two modern models (MaxViT-T and ConvNeXt-T) are clearly ahead of ResNet50. MaxViT-T has the best
  mean kappa and the best referable AUC, but its gap to ConvNeXt-T is smaller than the seed-to-seed spread,
  so I would not call it a win over ConvNeXt-T.
- MaxViT-T is also the slowest to train, about 364 seconds a run against 201 for ConvNeXt-T.
- Ensembling did not beat MaxViT-T on its own in kappa. The CNN plus hybrid pair gave the best accuracy
  and macro F1, by a small margin.

### Head and preprocessing (MaxViT-T)

| variant | kappa | accuracy | macro F1 | referable AUC |
|---|---|---|---|---|
| class logits (baseline) | 0.816 +/- 0.013 | 0.681 | 0.562 | 0.968 |
| ordinal regression | 0.834 +/- 0.019 | 0.652 | 0.516 | 0.973 |
| Ben Graham preprocessing | 0.827 +/- 0.020 | 0.702 | 0.564 | 0.971 |

Both changes nudge kappa up by about one standard deviation, which with 3 seeds is not enough to say they
help. The ordinal head also gives up exact-grade accuracy and macro F1 for a better kappa, which is what you
would expect from a loss that cares about distance and not about the exact class.

### GAN recipes

Conditional GAN at 64 by 64, 600 epochs. Recipe A copies the old notebook's settings (BCE loss, dropout
0.3, no augmentation, learning rate 1e-4) but is conditional and 64 pixels, so it is comparable and not a
reproduction. B uses hinge loss and spectral norm. C adds DiffAugment. D adds an EMA copy of the generator.

![gan fid](docs/gan_fid.png)

| recipe | FID | nearest training image, generated | nearest training image, unseen real |
|---|---|---|---|
| A original | 122.3 | 7.53 | 6.73 |
| B hinge + spectral norm | 117.8 | 8.12 | 6.73 |
| C + DiffAugment | 146.1 | 10.77 | 6.73 |
| D + EMA | 147.6 | 10.77 | 6.73 |

Read this one with care. FID here uses torchvision's InceptionV3 features, so it is only good for comparing
recipes with each other, not against published numbers. It is one seed per recipe, and the ordering is not
stable: I ran this stage three times (different machines and library versions) and recipe A came out at
134, 147 and 122, recipe C at 165 and 146, and recipe D at 115 and 148. Gaps between most recipes are about
as large as that run-to-run spread. The only pattern that held in all three runs is that B scored lower
than A, but by anything from 4 to 29 points. I would not claim more than that.

The distance columns are a memorisation check. Generated images sit further from their closest training image
than held-out real images do, so I see no sign of the generator copying training images.

### Does GAN data help a classifier

ResNet18 at 64 by 64, 5 seeds. The GAN images come from the recipe with the lowest FID (B in this run) and top up the small classes until they
match the biggest one.

![synthetic augmentation](docs/synthetic_augmentation.png)

| training data | kappa | accuracy | macro F1 | recall for grades 0 to 4 |
|---|---|---|---|---|
| real only | 0.564 +/- 0.039 | 0.531 | 0.391 | 0.73 / 0.18 / 0.50 / 0.42 / 0.14 |
| real + GAN images | 0.604 +/- 0.036 | 0.538 | 0.394 | 0.73 / 0.19 / 0.57 / 0.35 / 0.18 |
| real, class-balanced sampling | 0.598 +/- 0.028 | 0.496 | 0.401 | 0.67 / 0.46 / 0.37 / 0.31 / 0.30 |

Adding GAN images raised kappa by 0.04, about one standard deviation, and plain class-balanced sampling got
almost the same kappa with no GAN at all. In an earlier run, where a different recipe (D) had the lowest FID, the gain was
only 0.01. So I cannot show that these GAN images help beyond resampling. Class-balanced sampling is the one
that clearly moves the recall of the rare grades (mild from 0.18 to 0.46, proliferative from 0.14 to 0.30),
and it pays for that in overall accuracy.

## Limits

- 400 test images and 3 seeds means wide error bars. Differences under about 0.02 kappa should be treated as
  noise.
- One data split. No cross-validation.
- Hyperparameters were not tuned per model. The same recipe was used for every backbone.
- The GAN study is one seed per recipe and an uncalibrated FID.
- RetinaMNIST images are small and already compressed. These numbers say nothing about performance on
  full-resolution clinical photos, and none of this is a medical device.

## Running it

```
pip install -r requirements.txt
python -m retinolab.run_all --stage all --out runs
```

You need a GPU. The full run took about 90 minutes on two T4s with the classifier stages on one card and the
GAN stages on the other. Stages can be run one by one with `--stage cls|ens|abl|gan|synth|explain|report`, and
finished runs are skipped if you start again. If you run stages in separate processes, download the data once
first, because two processes downloading at the same moment corrupt the file.

```
pip install -r requirements-dev.txt
pytest
```

The tests run on CPU and cover the metrics, data handling, models, resuming, ensembles, the GAN and the FID.
`tools/plot_from_results.py` redraws the charts above from `results/RESULTS.md`.

## Layout

```
retinolab/   the package (data, models, training, GAN, FID, reports)
tests/       cpu tests
results/     RESULTS.md with all tables
docs/        charts used in this readme
tools/       small helper scripts
legacy/      the original notebook, unchanged
```

## License

MIT
