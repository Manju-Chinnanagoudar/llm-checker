from llm_checker.hardware import get_hardware_profile, HardwareProfile


def test_profile_returns_expected_types():
    profile = get_hardware_profile()
    assert isinstance(profile, HardwareProfile)
    assert isinstance(profile.total_ram_gb, float)
    assert isinstance(profile.available_ram_gb, float)
    assert isinstance(profile.cpu_cores, int)
    assert isinstance(profile.gpus, list)
    assert isinstance(profile.free_disk_gb, float)


def test_ram_is_sane():
    profile = get_hardware_profile()
    # anything under 0.5 GB or over 2TB is probably a bug
    assert 0.5 < profile.total_ram_gb < 2000
    assert profile.available_ram_gb <= profile.total_ram_gb


def test_os_is_known():
    profile = get_hardware_profile()
    assert profile.os in ("linux", "macos", "windows")


def test_cpu_cores_nonzero():
    profile = get_hardware_profile()
    assert profile.cpu_cores >= 1


def test_disk_is_sane():
    profile = get_hardware_profile()
    assert profile.free_disk_gb > 0
