import pandas as pd
import pytest

from applyagent.database import init_db
from applyagent.discovery import jobspy


def test_location_config_supports_shipped_and_legacy_format():
    assert jobspy._load_location_config({"location": {"accept_patterns": ["US"], "reject_patterns": ["UK"]}}) == (["US"], ["UK"])
    assert jobspy._load_location_config({"location_accept": ["CA"]}) == (["CA"], [])


@pytest.mark.parametrize("location,accept,reject,expected", [
    ("San Francisco", [], [], True),
    ("London", ["California"], [], False),
    ("Remote UK", ["Remote"], ["UK"], False),
    (None, ["US"], [], True),
])
def test_location_filter(location, accept, reject, expected):
    assert jobspy._location_ok(location, accept, reject) is expected


def test_board_failure_keeps_results_and_reports_error(monkeypatch, tmp_path):
    conn = init_db(tmp_path / "jobs.db")
    monkeypatch.setattr(jobspy, "get_connection", lambda: conn)
    calls = []

    def scrape(kwargs, **unused):
        calls.append(kwargs)
        if kwargs["site_name"] == ["linkedin"]:
            raise RuntimeError("unavailable")
        return pd.DataFrame([
            {"job_url": "https://example.org/1", "title": "Engineer", "location": "SF"},
            {"job_url": "https://example.org/2", "title": "Intern engineer", "location": "SF"},
        ])

    monkeypatch.setattr(jobspy, "_scrape_with_retry", scrape)
    args = ({"query": "engineer", "location": "SF"}, ["linkedin", "indeed"], 5, 72,
            None, {"country_indeed": "Canada"}, 0, [], [], {}, ["intern"])
    result = jobspy._run_one_search(*args)
    assert (result["errors"], result["new"], result["filtered"]) == (1, 1, 1)
    assert all(c["country_indeed"] == "Canada" for c in calls)
    second = jobspy._run_one_search(*args)
    assert (second["new"], second["existing"]) == (0, 1)


def test_boards_setting_is_used(monkeypatch):
    monkeypatch.setattr(jobspy, "_full_crawl", lambda **kwargs: kwargs)
    assert jobspy.run_discovery({"boards": ["glassdoor"]})["sites"] == ["glassdoor"]


def test_missing_fields_are_null_not_strings_or_remote(tmp_path):
    conn = init_db(tmp_path / "nulls.db")
    frame = pd.DataFrame([{"job_url": "https://example.org/job", "job_url_direct": None,
                           "location": None, "is_remote": float("nan")}])
    jobspy.store_jobspy_results(conn, frame, "linkedin")
    row = conn.execute("SELECT application_url, location FROM jobs").fetchone()
    assert tuple(row) == (None, None)


@pytest.mark.parametrize("location,remote,expected", [
    ("San Francisco, CA", True, True),
    ("San Francisco, CA", False, False),
    ("New York, NY", False, True),
    ("Albany, NY", False, False),
    (None, False, False),
    (None, True, True),
    ("Remote", False, True),
])
def test_strict_remote_or_nyc(location, remote, expected):
    assert jobspy._location_ok(location, ["New York, NY"], [], is_remote=remote,
                               allow_remote=True, accept_unknown=False) is expected
