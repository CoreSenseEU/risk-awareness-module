"""Smoke tests for riskam.provenance."""

from riskam.provenance import reproducibility_metadata


def test_metadata_has_expected_keys():
    md = reproducibility_metadata()
    for key in (
        "git_sha",
        "git_dirty",
        "hostname",
        "timestamp_utc",
        "python_version",
        "packages",
    ):
        assert key in md


def test_packages_dict_includes_tracked_names():
    md = reproducibility_metadata()
    for name in ("torch", "ultralytics", "numpy", "opencv-python"):
        assert name in md["packages"]


def test_timestamp_is_iso8601_utc():
    md = reproducibility_metadata()
    # Ends with '+00:00' per datetime.isoformat() with tz=utc.
    assert md["timestamp_utc"].endswith("+00:00")


def test_hostname_is_nonempty_string():
    md = reproducibility_metadata()
    assert isinstance(md["hostname"], str)
    assert len(md["hostname"]) > 0
