from ecg_mi.config import load_yaml


def test_load_yaml_reads_repo_configs():
    cohorts = load_yaml("configs/cohorts.yaml")
    paths = load_yaml("configs/paths.yaml")

    assert cohorts["dataset_d"]["n_certain"] == 597
    assert paths["cohorts_dir"] == "data/cohorts"
