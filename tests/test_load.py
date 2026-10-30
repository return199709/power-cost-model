import numpy as np
import pandas as pd
import pytest

from dcop.io import load_sites
from dcop.load import delivered_and_unserved, parse_period, ramp_mw, site_load_kw, synthetic_shape
from dcop.schemas import Site


def make_site(**kw) -> Site:
    base = dict(
        site_id="S", name="S", region="R", utility="U", load_mw_peak=100.0,
        load_mw_ramp=[("2027-Q1", 25.0), ("2027-Q3", 50.0), ("2028", 100.0)],
        load_factor=0.9, ambient_c_design=30.0, elevation_m=100.0,
    )
    base.update(kw)
    return Site(**base)


def test_parse_period():
    assert parse_period("2028-Q3") == pd.Timestamp("2028-07-01")
    assert parse_period("2029") == pd.Timestamp("2029-01-01")
    with pytest.raises(ValueError):
        parse_period("Q1-2028")


def test_ramp_is_a_step_function_by_quarter():
    idx = pd.DatetimeIndex(["2026-12-31", "2027-01-01", "2027-06-30", "2027-07-01", "2028-01-01"])
    assert ramp_mw(make_site(), idx).tolist() == [0.0, 25.0, 25.0, 50.0, 100.0]


def test_synthetic_shape_peaks_at_one_with_annual_mean_equal_to_load_factor():
    idx = pd.date_range("2027-01-01", "2028-01-01", freq="h", inclusive="left")
    shape = synthetic_shape(0.85, idx)
    assert shape.max() == pytest.approx(1.0)
    assert shape.min() >= 0.0
    assert shape.mean() == pytest.approx(0.85, abs=0.002)


def test_synthetic_shape_needs_a_profile_for_low_load_factor():
    with pytest.raises(ValueError, match="load_profile_ref"):
        synthetic_shape(0.4, pd.date_range("2027-01-01", periods=24, freq="h"))


def test_site_load_kw_covers_whole_years_and_scales_by_ramp():
    load = site_load_kw(make_site(), [2027, 2028])
    assert load.index[0] == pd.Timestamp("2027-01-01 00:00")
    assert load.index[-1] == pd.Timestamp("2028-12-31 23:00")
    assert load["2027-01"].max() <= 25_000 + 1e-6
    assert load["2028"].max() == pytest.approx(100_000.0)


def test_profile_from_csv_inside_data_dir(tmp_path):
    pd.DataFrame({"shape": np.full(8760, 0.5)}).to_csv(tmp_path / "profile.csv", index=False)
    site = make_site(load_profile_ref="profile.csv", load_mw_ramp=[("2027", 100.0)])
    load = site_load_kw(site, [2027], data_dir=tmp_path)
    assert load.mean() == pytest.approx(50_000.0)


def test_profile_ref_cannot_escape_data_dir(tmp_path):
    (tmp_path / "data").mkdir()
    pd.DataFrame({"shape": np.full(8760, 0.5)}).to_csv(tmp_path / "outside.csv", index=False)
    site = make_site(load_profile_ref="../outside.csv", load_mw_ramp=[("2027", 100.0)])
    with pytest.raises(ValueError, match="escapes"):
        site_load_kw(site, [2027], data_dir=tmp_path / "data")


def test_delivered_and_unserved_balance():
    load = pd.Series([100.0, 200.0])
    delivered, unserved = delivered_and_unserved(load, availability=0.9, curtailment=0.1)
    assert delivered.tolist() == pytest.approx([81.0, 162.0])
    assert (delivered + unserved).tolist() == pytest.approx(load.tolist())


@pytest.mark.parametrize(
    "ramp",
    [
        [("2028", 50.0), ("2027", 60.0)],  # out of order
        [("2027", 60.0), ("2028", 50.0)],  # decreasing
        [("2027", 150.0)],  # above peak
        [],
    ],
)
def test_site_rejects_bad_ramps(ramp):
    with pytest.raises(ValueError):
        make_site(load_mw_ramp=ramp)


def test_ids_with_dots_rejected():
    with pytest.raises(ValueError, match="'.'"):
        make_site(site_id="A.B")
