from navigen_safety.checks import DebouncedThreshold, StaleMonitor, command_is_valid


def test_command_validity():
    assert command_is_valid(0.2, -0.5)
    assert not command_is_valid(float('nan'), 0.0)
    assert not command_is_valid(0.0, float('inf'))
    assert not command_is_valid(float('-inf'), float('nan'))


def test_debounce_ignores_single_noisy_reading():
    us = DebouncedThreshold(threshold=0.35, consecutive=3)
    assert not us.update(0.1)   # 1st low reading: no trigger yet
    assert not us.update(2.0)   # recovered -> counter reset
    assert not us.update(0.1)
    assert not us.update(0.1)
    assert us.update(0.1)       # 3 consecutive low readings -> trigger


def test_debounce_invalid_readings_keep_state():
    us = DebouncedThreshold(threshold=0.35, consecutive=2)
    us.update(0.1)
    us.update(0.1)
    assert us.triggered
    assert us.update(-1.0)      # invalid reading does not clear the trigger
    assert us.update(2.0) or not us.triggered  # valid far reading clears it


def test_stale_monitor():
    mon = StaleMonitor(timeout_s=0.5)
    assert mon.is_stale(0.0)          # never updated -> stale
    mon.update(10.0)
    assert not mon.is_stale(10.4)
    assert mon.is_stale(10.6)
