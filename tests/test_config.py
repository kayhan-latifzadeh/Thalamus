"""Study configs, and failing fast when they are wrong.

A config error must surface before anything starts listening. Discovering that a
seed is a string forty minutes into a dry run is the exact waste of time the toolkit
exists to prevent.
"""

from __future__ import annotations

import pytest

from thalamus.config import ConfigError, StudyConfig
from thalamus.devices import ReplayDevice, SyntheticDevice


def write(tmp_path, text, name="study.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestLoading:
    def test_loads_a_study(self, tmp_path):
        path = write(
            tmp_path,
            """
            core:
              device_port: 9100
              client_port: 9101
            devices:
              - id: eye
                type: synthetic
                rate: 150
                signals:
                  pupil: {kind: sine, freq: 0.2}
            """,
        )
        config = StudyConfig.load(path)
        assert config.device_port == 9100
        assert [d.id for d in config.devices] == ["eye"]

        device = config.devices[0].build(host="localhost", port=9100)
        assert isinstance(device, SyntheticDevice)
        assert device.rate == 150

    def test_json_works_too(self, tmp_path):
        path = write(
            tmp_path,
            '{"devices": [{"id": "x", "type": "synthetic", "rate": 10, '
            '"signals": {"a": {"kind": "sine"}}}]}',
            name="study.json",
        )
        assert len(StudyConfig.load(path).devices) == 1

    def test_defaults_apply_when_core_is_omitted(self, tmp_path):
        config = StudyConfig.load(write(tmp_path, "devices: []"))
        assert (config.device_port, config.client_port) == (9000, 9001)

    def test_a_relative_data_path_resolves_against_the_config_not_the_cwd(self, tmp_path):
        # Otherwise a study file only works when run from one particular directory.
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "eeg.csv").write_text("timestamp,Fp1\n1000,1.5\n", encoding="utf-8")

        path = write(
            tmp_path,
            "devices:\n  - id: eeg\n    type: replay\n    path: data/eeg.csv\n    rate: 250\n",
        )
        config = StudyConfig.load(path)
        device = config.devices[0].build(host="localhost", port=9000)
        assert isinstance(device, ReplayDevice)
        assert device.path.is_absolute()


class TestValidation:
    def test_a_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="no such config"):
            StudyConfig.load(tmp_path / "nope.yaml")

    def test_a_device_needs_an_id_and_a_type(self):
        with pytest.raises(ConfigError, match="has no 'id'"):
            StudyConfig.from_dict({"devices": [{"type": "synthetic"}]})
        with pytest.raises(ConfigError, match="has no 'type'"):
            StudyConfig.from_dict({"devices": [{"id": "x"}]})

    def test_an_unknown_device_type_lists_the_known_ones(self):
        with pytest.raises(ConfigError, match="replay"):
            StudyConfig.from_dict({"devices": [{"id": "x", "type": "telepathy"}]})

    def test_duplicate_device_ids_are_rejected(self):
        # Two devices sharing an id would silently interleave into one stream.
        with pytest.raises(ConfigError, match="duplicate"):
            StudyConfig.from_dict(
                {
                    "devices": [
                        {"id": "x", "type": "synthetic", "rate": 1, "signals": {"a": {}}},
                        {"id": "x", "type": "synthetic", "rate": 1, "signals": {"a": {}}},
                    ]
                }
            )

    def test_a_bad_stage_is_caught_at_load_time_not_at_run_time(self):
        with pytest.raises(ConfigError, match="simulate"):
            StudyConfig.from_dict(
                {
                    "devices": [
                        {
                            "id": "x",
                            "type": "synthetic",
                            "rate": 1,
                            "signals": {"a": {}},
                            "simulate": [{"stage": "does_not_exist"}],
                        }
                    ]
                }
            )

    def test_a_typo_at_the_top_level_is_not_silently_ignored(self):
        # "devcies:" should be an error, not a study that quietly runs no devices.
        with pytest.raises(ConfigError, match="unknown top-level"):
            StudyConfig.from_dict({"devcies": []})

    def test_a_bad_device_option_names_the_device(self):
        config = StudyConfig.from_dict(
            {"devices": [{"id": "eeg", "type": "replay", "path": "/nowhere/eeg.csv"}]}
        )
        with pytest.raises(ConfigError, match="eeg"):
            config.devices[0].build(host="localhost", port=9000)
