"""HTTP payload → RunCleanupRequest: overrides and new config fields pass through."""
import json
from pathlib import Path

from videoclean.application.use_cases.manage_jobs import cleanup_request_from_row, pipeline_config_from_dict


class Row(dict):
    def __getitem__(self, k):
        return dict.__getitem__(self, k)


def _row(payload: dict) -> Row:
    return Row({
        "id": "j1",
        "state": "QUEUED",
        "input_path": "/tmp/in.mp4",
        "output_path": "/tmp/out.mp4",
        "prompt": "p",
        "request_json": json.dumps(payload),
    })


def test_overrides_pass_through():
    req = cleanup_request_from_row(_row({
        "targets_override": [{"kind": "object", "query": "mug"}],
        "tracks_override": [{"id": 0, "label": "logo", "boxes": [[1, 1, 4, 4]]}],
        "masks_override": ["/tmp/m.png"],
        "keep_workdir": True,
    }))
    assert req.targets_override == [{"kind": "object", "query": "mug"}]
    assert req.tracks_override == [{"id": 0, "label": "logo", "boxes": [[1, 1, 4, 4]]}]
    assert req.masks_override == [Path("/tmp/m.png")]
    assert req.keep_workdir is True


def test_no_overrides_means_none():
    req = cleanup_request_from_row(_row({}))
    assert req.targets_override is None
    assert req.tracks_override is None
    assert req.masks_override is None


def test_serialize_new_fields():
    from server.service import serialize_clean_form

    out = serialize_clean_form({
        "llm_base_url": " https://api.example.com/v1 ",
        "llm_api_key": " sk-test ",
        "keep_workdir": "1",
        "min_mask_coverage": "0.01",
        "verify_max_coverage": "0.2",
        "formats": "mp4,webm",
    })
    assert out["llm_base_url"] == "https://api.example.com/v1"
    assert out["llm_api_key"] == "sk-test"
    assert out["keep_workdir"] is True
    assert out["min_mask_coverage"] == 0.01
    assert out["verify_max_coverage"] == 0.2
    assert out["formats"] == ["mp4", "webm"]


def test_serialize_defaults():
    from server.service import serialize_clean_form

    out = serialize_clean_form({})
    assert out["formats"] == ["mp4"]
    assert out["min_mask_coverage"] == 0.0004
    assert out["verify_max_coverage"] == 0.12
    assert out["keep_workdir"] is False
    assert out["llm_base_url"] == ""
