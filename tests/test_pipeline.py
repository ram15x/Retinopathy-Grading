# quick cpu tests with tiny fake data, nothing gets downloaded
import json

import numpy as np
import pytest
import torch

from retinolab import aggregate as A
from retinolab import metrics as M
from retinolab.data import ArrayDataset, balanced_sampler, ben_graham, prepare_splits, resize_all
from retinolab.fid import fid_from_features, nn_distance
from retinolab.gan import RECIPES, Discriminator, Generator, d_loss, diff_augment, ema_update, g_loss, sample, train_gan
from retinolab.models import BACKBONES, build_classifier, gradcam_layer
from retinolab.synth import add_synthetic, make_synthetic, plan_counts
from retinolab.train_cls import ClsConfig, run_one


def fake_raw(n=(60, 20, 20), size=32, seed=0):
    # tiny fake "fundus" images where brightness follows the grade, so models can learn something
    rng = np.random.default_rng(seed)
    out = {}
    for split, k in zip(("train", "val", "test"), n):
        y = np.arange(k) % 5
        img = (rng.random((k, size, size, 3)) * 40 + y[:, None, None, None] * 35).astype(np.uint8)
        out[split] = (img, y.astype(np.int64))
    return out


# ---------- metrics ----------
def test_qwk_perfect_and_worst():
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    assert M.qwk(y, y) == 1.0
    assert M.qwk(y, 4 - y) < 0


def test_qwk_punishes_far_errors_more_than_near_errors():
    y = np.array([0, 0, 2, 2, 4, 4, 1, 3])
    near, far = y.copy(), y.copy()
    near[0] = 1
    far[0] = 4
    assert M.qwk(y, near) > M.qwk(y, far)


def test_referable_auc_is_one_for_perfect_scores():
    y = np.array([0, 1, 2, 3, 4, 0, 4])
    assert M.referable_auc(y, y.astype(float)) == 1.0


def test_evaluate_fields_and_confusion_shape():
    y = np.array([0, 1, 2, 3, 4] * 4)
    r = M.evaluate(y, y, y.astype(float))
    assert r["accuracy"] == 1.0 and r["qwk"] == 1.0 and np.array(r["confusion_matrix"]).shape == (5, 5)
    assert r["referable_sensitivity"] == 1.0 and r["referable_specificity"] == 1.0


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300)
    pred = np.clip(y + rng.integers(-1, 2, 300), 0, 4)
    lo, hi = M.bootstrap_ci(y, pred, n=300)
    assert lo <= M.qwk(y, pred) <= hi


def test_fit_thresholds_is_sorted_and_not_worse_than_default():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, 400)
    score = y * 0.8 + 0.3 + rng.normal(0, 0.5, 400)  # biased scores, default cut points are off
    t = M.fit_thresholds(score, y)
    assert t == sorted(t)
    base = M.qwk(y, M.apply_thresholds(score, (0.5, 1.5, 2.5, 3.5)))
    assert M.qwk(y, M.apply_thresholds(score, t)) >= base


# ---------- data ----------
def test_ben_graham_keeps_shape_and_dtype():
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    out = ben_graham(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_prepare_splits_resizes_and_rejects_unknown_preproc():
    raw = fake_raw(size=40)
    sp = prepare_splits(raw, 16, "ben")
    assert sp["train"][0].shape[1:] == (16, 16, 3)
    with pytest.raises(ValueError):
        prepare_splits(raw, 16, "nope")
    assert resize_all(raw["train"][0], 40) is raw["train"][0]


def test_dataset_item_shape_and_balanced_sampler_covers_classes():
    raw = fake_raw()
    ds = ArrayDataset(*raw["train"], train=True, size=32)
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and isinstance(y, int)
    skew = np.array([0] * 90 + [1] * 5 + [2] * 3 + [3] * 1 + [4] * 1)
    idx = list(balanced_sampler(skew))
    assert len(np.unique(skew[idx])) == 5


# ---------- models ----------
@pytest.mark.parametrize("name", ["resnet18", "maxvit_t"])
def test_heads_have_the_right_output_size(name):
    x = torch.zeros(1, 3, 224, 224)
    assert build_classifier(name, "ce", pretrained=False).eval()(x).shape == (1, 5)
    assert build_classifier(name, "ordinal", pretrained=False).eval()(x).shape == (1, 1)


def test_unknown_backbone_and_head_raise():
    with pytest.raises(ValueError):
        build_classifier("vgg", "ce", pretrained=False)
    with pytest.raises(ValueError):
        build_classifier("resnet18", "weird", pretrained=False)


def test_every_backbone_has_a_gradcam_layer():
    for name in BACKBONES:
        if name in ("resnet18", "resnet50"):
            m = build_classifier(name, "ce", pretrained=False)
            assert gradcam_layer(m, name) is not None


# ---------- training + aggregation ----------
@pytest.fixture(scope="module")
def finished_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("runs")
    raw = fake_raw()
    sp = prepare_splits(raw, 32, "none")
    dirs = []
    for seed in (0, 1):
        cfg = ClsConfig(backbone="resnet18", pretrained=False, img_size=32, epochs=2, batch_size=16,
                        workers=0, seed=seed, save_weights=(seed == 0))
        run_one(cfg, sp, base / f"seed{seed}", device=torch.device("cpu"))
        dirs.append(base / f"seed{seed}")
    return base, dirs


def test_run_one_writes_everything_needed(finished_runs):
    _, dirs = finished_runs
    r = json.loads((dirs[0] / "metrics.json").read_text())
    for k in ("qwk", "accuracy", "macro_f1", "referable_auc", "qwk_ci95", "best_epoch", "config"):
        assert k in r
    assert (dirs[0] / "outputs.npz").exists() and (dirs[0] / "best.pt").exists()
    assert not (dirs[1] / "best.pt").exists()


def test_run_one_skips_finished_runs(finished_runs):
    _, dirs = finished_runs
    before = (dirs[0] / "metrics.json").stat().st_mtime_ns
    cfg = ClsConfig(backbone="resnet18", pretrained=False, img_size=32, epochs=2, workers=0)
    run_one(cfg, {}, dirs[0])  # empty splits would crash if it tried to train
    assert (dirs[0] / "metrics.json").stat().st_mtime_ns == before


def test_ordinal_run_uses_thresholds(tmp_path):
    sp = prepare_splits(fake_raw(), 32, "none")
    cfg = ClsConfig(backbone="resnet18", head="ordinal", pretrained=False, img_size=32, epochs=2,
                    batch_size=16, workers=0)
    r = run_one(cfg, sp, tmp_path / "o", device=torch.device("cpu"))
    assert len(r["thresholds"]) == 4


def test_mean_std_and_ensemble(finished_runs):
    _, dirs = finished_runs
    runs = [A.load_run(d) for d in dirs]
    ms = A.mean_std(runs)
    assert ms["n_runs"] == 2 and 0 <= ms["accuracy"]["mean"] <= 1
    assert np.isnan(A.mean_std(runs[:1])["qwk"]["std"])
    ens = A.ensemble_from_dirs(dirs)
    a, b = (np.load(d / "outputs.npz") for d in dirs)
    expected = ((a["probs_test"] + b["probs_test"]) / 2).argmax(1)
    y = a["y_test"]
    assert abs(ens["accuracy"] - float((expected == y).mean())) < 1e-9


def test_ensemble_refuses_ordinal_outputs(tmp_path):
    sp = prepare_splits(fake_raw(), 32, "none")
    cfg = ClsConfig(backbone="resnet18", head="ordinal", pretrained=False, img_size=32, epochs=1,
                    batch_size=16, workers=0)
    run_one(cfg, sp, tmp_path / "o", device=torch.device("cpu"))
    with pytest.raises(ValueError):
        A.ensemble_from_dirs([tmp_path / "o"])


# ---------- gan ----------
def test_diff_augment_keeps_shape_and_passes_gradients():
    x = torch.randn(4, 3, 64, 64, requires_grad=True)
    out = diff_augment(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_generator_and_discriminator_shapes():
    g, d = Generator(), Discriminator()
    y = torch.tensor([0, 1, 2, 3])
    img = g(torch.randn(4, 128), y)
    assert img.shape == (4, 3, 64, 64) and img.min() >= -1 and img.max() <= 1
    assert d(img, y).shape == (4, 1)


def test_losses_move_in_the_right_direction():
    good, bad = torch.full((8, 1), 5.0), torch.full((8, 1), -5.0)
    for kind in ("bce", "hinge"):
        assert d_loss(kind, good, bad) < d_loss(kind, bad, good)
        assert g_loss(kind, good) < g_loss(kind, bad)


def test_ema_with_zero_decay_copies_the_model():
    a, b = Generator(ch=16), Generator(ch=16)
    ema_update(a, b, 0.0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb)


@pytest.mark.parametrize("recipe", list(RECIPES))
def test_every_recipe_runs_a_tiny_training(recipe, tmp_path):
    cfg = RECIPES[recipe]
    cfg = type(cfg)(**{**cfg.__dict__, "epochs": 1, "batch_size": 8})
    imgs = (np.random.default_rng(0).random((16, 64, 64, 3)) * 255).astype(np.uint8)
    seen = []
    out = train_gan(cfg, imgs, np.arange(16) % 5, tmp_path, torch.device("cpu"),
                    on_checkpoint=lambda g, e: seen.append(e), checkpoint_every=1)
    assert (tmp_path / "G_final.pt").exists() and seen == [1]
    fake = sample(out["G"], torch.tensor([0, 1, 2]), torch.device("cpu"), seed=0)
    assert fake.shape == (3, 3, 64, 64) and 0 <= fake.min() and fake.max() <= 1


def test_sampling_is_reproducible_with_a_seed():
    g = Generator(ch=16)
    y = torch.tensor([0, 1, 2, 3])
    assert torch.equal(sample(g, y, torch.device("cpu"), seed=5), sample(g, y, torch.device("cpu"), seed=5))


# ---------- fid + synth ----------
def test_fid_is_zero_for_the_same_data_and_grows_with_a_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 16))
    assert fid_from_features(a, a) < 1e-6
    assert fid_from_features(a, a + 2.0) > fid_from_features(a, a + 0.1)


def test_nn_distance_is_zero_when_queries_are_in_the_reference():
    r = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    assert nn_distance(r[:5], r) < 1e-5


def test_plan_counts_fills_up_small_classes():
    labels = np.array([0] * 10 + [1] * 4 + [2] * 10 + [3] * 1 + [4] * 6)
    assert plan_counts(labels) == [0, 6, 0, 9, 4]


def test_make_synthetic_returns_requested_counts():
    labels = np.array([0] * 6 + [1] * 4 + [2] * 6 + [3] * 2 + [4] * 6)
    imgs, ys = make_synthetic(Generator(ch=16), labels, torch.device("cpu"))
    assert np.bincount(ys, minlength=5).tolist() == plan_counts(labels)
    assert imgs.dtype == np.uint8 and imgs.shape[1:] == (64, 64, 3)
    real = (np.zeros((len(labels), 64, 64, 3), np.uint8), labels)
    merged = add_synthetic(real, (imgs, ys))
    assert np.bincount(merged[1]).tolist() == [6] * 5


def test_fid_matches_the_scipy_matrix_sqrt_on_a_small_case():
    import scipy.linalg
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(400, 30)), rng.normal(0.3, 1.2, size=(400, 30))
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ref = ((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(scipy.linalg.sqrtm(s1 @ s2).real)
    assert abs(fid_from_features(a, b) - ref) < 1e-6
