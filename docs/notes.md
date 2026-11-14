# Notes on the results

- This is research and learning code and not a medical device. RetinaMNIST images are small and already compressed, so the numbers say nothing about full-resolution clinical photos.
- 400 test images and 3 seeds mean wide error bars. Differences under about 0.02 kappa are noise.
- There is one data split and no cross-validation. Hyperparameters were not tuned per model, the same recipe was used for every backbone.
- MaxViT-T has the best mean kappa, but its gap to ConvNeXt-T is smaller than the spread between seeds. The ordinal head and Ben Graham preprocessing each add about one standard deviation of kappa, which with 3 seeds is not enough to say they help.
- The GAN study is one seed per recipe with an uncalibrated FID (torchvision InceptionV3 features), so it only ranks recipes against each other. The stage was run three times on different machines and library versions, and the ordering moved: recipe A scored 134, 147 and 122, C scored 165 and 146, D scored 115 and 148. The only result that held every time is that B scored lower than A, by 4 to 29 points.
- GAN images raised kappa by 0.04, about one standard deviation, and class-balanced sampling got almost the same without a GAN. In an earlier run, where recipe D had the lowest FID, the gain was only 0.01. Class-balanced sampling is what clearly moves recall for the rare grades, and it costs overall accuracy.
- If you run stages in separate processes, download the data once first. Two processes downloading at the same moment corrupt the file.
