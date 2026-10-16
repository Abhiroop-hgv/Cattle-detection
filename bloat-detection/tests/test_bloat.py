import numpy as np
from bloat.geometry import measure, Config
from bloat.synthetic import synthetic_cow
from bloat.baseline import CowBaseline
from bloat.monitor import (BloatMonitor, Burst, summarize_burst, OK, WATCH, URGENT, UNKNOWN)


def test_normal_cow_is_symmetric():
    m = measure(synthetic_cow())
    assert m.valid
    assert abs(m.asymmetry) < 0.02


def test_left_swelling_gives_positive_asymmetry():
    normal = measure(synthetic_cow())
    swollen = measure(synthetic_cow(swell_left=0.15))
    assert swollen.asymmetry > normal.asymmetry + 0.03
    assert swollen.flank_to_hip > normal.flank_to_hip


def test_right_swelling_gives_negative_asymmetry():
    assert measure(synthetic_cow(swell_right=0.15)).asymmetry < -0.03


def test_mirror_flag_flips_sides():
    swollen = synthetic_cow(swell_left=0.15)
    flipped = measure(swollen, Config(left_is_image_left=False))
    assert flipped.asymmetry < -0.03


def test_distance_does_not_change_ratios():
    near = measure(synthetic_cow(scale=1.0))
    far = measure(synthetic_cow(scale=0.6))
    assert abs(near.flank_to_hip - far.flank_to_hip) < 0.06


def test_invalid_cases():
    assert not measure(np.zeros((480, 360), bool)).valid
    cut = synthetic_cow()
    cut[:, :60] = False
    cut[200, 0] = True  # touches the edge
    assert not measure(cut).valid
    assert not measure(synthetic_cow(scale=0.2)).valid


def test_burst_needs_enough_good_frames():
    good = measure(synthetic_cow())
    bad = measure(np.zeros((10, 10), bool))
    assert not summarize_burst([good, bad, bad]).valid
    assert summarize_burst([good, good, good]).valid


def _burst(asym, ratio=1.0):
    return Burst(True, {"asymmetry": asym, "flank_to_hip": ratio}, n_valid=5)


def _trained_baseline(rng=np.random.default_rng(1)):
    b = CowBaseline()
    for _ in range(30):
        b.add(18, {"asymmetry": rng.normal(0.0, 0.01), "flank_to_hip": rng.normal(1.0, 0.02)})
    return b


def test_unknown_when_not_assessable():
    mon = BloatMonitor(_trained_baseline())
    assert mon.update(18, Burst(False, reason="few frames")).level == UNKNOWN


def test_normal_reading_is_ok_and_learned():
    mon = BloatMonitor(_trained_baseline())
    assert mon.update(18, _burst(0.005)).level == OK


def test_big_jump_is_watch_then_urgent():
    mon = BloatMonitor(_trained_baseline())
    assert mon.update(18, _burst(0.08, 1.15)).level == WATCH
    assert mon.update(18, _burst(0.20, 1.40)).level == URGENT


def test_rising_trend_becomes_urgent():
    mon = BloatMonitor(_trained_baseline())
    levels = [mon.update(18, _burst(a, 1.0 + a)).level for a in (0.06, 0.07, 0.09)]
    assert levels[0] == WATCH and levels[-1] == URGENT


def test_sick_readings_do_not_pollute_baseline():
    base = _trained_baseline()
    mon = BloatMonitor(base)
    before = len(base._data[6]["asymmetry"])
    mon.update(18, _burst(0.2, 1.4))
    assert len(base._data[6]["asymmetry"]) == before


def test_no_baseline_uses_fixed_limits():
    mon = BloatMonitor(CowBaseline())
    a = mon.update(18, _burst(0.07))
    assert a.level == WATCH and any("no baseline" in r for r in a.reasons)
