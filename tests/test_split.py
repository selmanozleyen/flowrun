import pandas as pd
import pytest

from flowrun.split import shared_source_combination, shared_source_holdout

OBS = pd.DataFrame(
    {
        "cell_line": ["A"] * 6 + ["B"] * 6,
        "drug": ["control", "d0", "d1", "d2", "d3", "d4"] * 2,
    }
)
IS_SOURCE = (OBS["drug"] == "control").to_numpy()


def test_holdout_by_cell_line_shares_sources():
    labels = shared_source_holdout(OBS, "cell_line", is_source=IS_SOURCE, test=["B"])
    assert labels.tolist() == ["all"] + ["train"] * 5 + ["all"] + ["test"] * 5


def test_holdout_rejects_unknown_values():
    with pytest.raises(ValueError, match="not found"):
        shared_source_holdout(OBS, "cell_line", is_source=IS_SOURCE, val=["C"])


def test_combination_keeps_every_cell_line_in_train():
    labels = shared_source_combination(
        OBS,
        ["cell_line", "drug"],
        is_source=IS_SOURCE,
        always_train_keys=["cell_line"],
        test_fraction=0.4,
        val_fraction=0.2,
    )
    per_line = pd.crosstab(OBS["cell_line"], labels)
    assert (per_line["test"] == 2).all() and (per_line["val"] == 1).all() and (per_line["train"] == 2).all()
    assert (labels[IS_SOURCE] == "all").all()
