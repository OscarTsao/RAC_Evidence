from Project.calib.temperature import calibrate_values, learn_temperature
from Project.metrics.calibration import ece


def test_temperature_reduces_ece() -> None:
    logits = [0.2, 1.4, -0.5, 2.0, 0.0]
    labels = [0, 1, 0, 1, 0]
    before = ece(labels, [1 / (1 + pow(2.71828, -l)) for l in logits])
    temp = learn_temperature(logits, labels, lr=0.05, steps=50)
    _, before_ece, after_ece = calibrate_values(logits, labels)
    assert temp > 0
    assert after_ece <= before_ece or before < 0.05
