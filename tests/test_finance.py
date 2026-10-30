import pytest

from dcop import finance


def test_annuity_payment_matches_hand_calc():
    # $1,000,000 at 8% over 15 years: 1e6 x 0.08 / (1 - 1.08^-15) = 116,829.5
    assert finance.annuity_payment(1_000_000, 0.08, 15) == pytest.approx(116_829.54, abs=0.01)


def test_annuity_zero_rate_is_straight_line():
    assert finance.annuity_payment(1_200.0, 0.0, 12) == pytest.approx(100.0)


def test_annuity_schedule_pv_recovers_principal():
    # First payment one year after the base year: PV at the base year equals principal.
    sched = finance.annuity_schedule(5_000_000, 0.08, 15, first_payment_year=2027)
    assert len(sched) == 15 and min(sched) == 2027 and max(sched) == 2041
    assert finance.npv(sched, 0.08, base_year=2026) == pytest.approx(5_000_000)


def test_npv_hand_calc():
    # 100/1.1 + 100/1.21 = 173.553719
    assert finance.npv({2027: 100.0, 2028: 100.0}, 0.10, 2026) == pytest.approx(173.5537, abs=1e-4)


def test_npv_base_year_flow_is_undiscounted():
    assert finance.npv({2026: 250.0}, 0.08, 2026) == pytest.approx(250.0)


def test_levelize_flat_cost_and_energy_gives_unit_price():
    cost = {y: 7_000_000.0 for y in range(2027, 2042)}
    mwh = {y: 100_000.0 for y in range(2027, 2042)}
    assert finance.levelize(cost, mwh, 0.08, 2026) == pytest.approx(70.0)


def test_levelize_weights_by_discounted_energy():
    # Year 1: $100 for 1 MWh, year 2: $100 for 3 MWh, r = 10%
    # NPV cost = 100/1.1 + 100/1.21 = 173.5537; NPV mwh = 1/1.1 + 3/1.21 = 3.38843
    out = finance.levelize({2027: 100.0, 2028: 100.0}, {2027: 1.0, 2028: 3.0}, 0.10, 2026)
    assert out == pytest.approx(173.5537 / 3.38843, rel=1e-5)


def test_levelize_rejects_zero_energy():
    with pytest.raises(ValueError):
        finance.levelize({2027: 1.0}, {2027: 0.0}, 0.08, 2026)


def test_escalate():
    assert finance.escalate(100.0, 0.02, 3) == pytest.approx(106.1208)


def test_to_real_deflates_future_and_inflates_past_dollars():
    assert finance.to_real(110.0, 2027, 2026, 0.10) == pytest.approx(100.0)
    assert finance.to_real(100.0, 2025, 2026, 0.10) == pytest.approx(110.0)
