import pandas as pd
import pytest

from flowrun.split import combination, holdout

OBS = pd.DataFrame(
    {
        "cell_line": ["A"] * 5 + ["B"] * 5,
        "drug": ["control", "d0", "d1", "d2", "d3", "control", "d0", "d1", "d2", "d3"],
    }
)


def test_holdout_by_cell_line():
    labels = holdout(OBS, "cell_line", test=["B"], control_key="drug")
    assert labels.tolist() == ["control"] + ["train"] * 4 + ["control"] + ["test"] * 4


def test_holdout_rejects_unknown_values():
    with pytest.raises(ValueError, match="not found"):
        holdout(OBS, "cell_line", val=["C"])


def test_combination_keeps_every_cell_line_in_train():
    labels = combination(
        OBS, ["cell_line", "drug"], always_train_keys=["cell_line"], control_key="drug", test_fraction=0.5
    )
    per_line = pd.crosstab(OBS["cell_line"], labels)
    assert (per_line["train"] == 2).all() and (per_line["test"] == 2).all() and (per_line["control"] == 1).all()
    assert combination(OBS, ["cell_line", "drug"], control_key="drug", seed=3).equals(
        combination(OBS, ["cell_line", "drug"], control_key="drug", seed=3)
    )
