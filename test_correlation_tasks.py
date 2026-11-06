import pytest
import pandas as pd
import numpy as np
from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput
COLS = ["Gender", "FSIQ", "VIQ", "PIQ", "Weight", "Height", "MRI_Count"]
def write_brain(tmp_path, rows):
    path = tmp_path / "brainsize.txt"
    pd.DataFrame(rows, columns=COLS).to_csv(path, sep='\t', index=False)
    return BrainDataInput(csv_path=str(path))

@pytest.mark.parametrize("rows,men,women", [
    ([], 0, 0),
    ([["Male", 90, 85, 80, 70, 170, 900000]], 1, 0),
    ([["Female", 90, 85, 80, 60, 160, 900000]], 0, 1),
    ([["Male", 90, 85, 80, 70, 170, 900000],
      ["Female", 90, 85, 80, 60, 160, 900000]], 1, 1),
])
def test_counts(tmp_path, rows, men, women):
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    assert s.men_count == men
    assert s.women_count == women

def test_only_men_women_corrs_are_nan(tmp_path):
    rows = [
        ["Male", 90, 85, 80, 70, 170, 900000],
        ["Male", 110, 105, 100, 80, 180, 1000000],
    ]
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    assert all(pd.isna(v) for v in s.women_mri_correlation.values())

def test_constant_column_gives_nan(tmp_path):
    rows = [
        ["Male", 90, 85, 80, 70, 170, 900000],
        ["Male", 100, 95, 90, 75, 170, 950000],
        ["Male", 110, 105, 100, 80, 170, 1000000],
    ]
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    assert pd.isna(s.men_mri_correlation["Height"])
    assert s.strongest_mri_feature != "Height"

def test_perfect_correlation(tmp_path):
    rows = [
        ["Male", 90, 85, 80, 70, 170, 100000],
        ["Male", 95, 90, 85, 72, 175, 200000],
        ["Male", 100, 95, 90, 74, 180, 300000],
    ]
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    assert s.men_mri_correlation["Height"] == pytest.approx(1.0, abs=1e-9)
    assert s.strongest_mri_feature == "Height"

def test_na_values_handled(tmp_path):
    rows = [
        ["Male", 90, "NA", 80, 70, 170, 900000],
        ["Male", 100, 95, 90, 75, 175, 950000],
        ["Male", 110, 105, 100, 80, 180, 1000000],
    ]
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    assert not pd.isna(s.men_mri_correlation["FSIQ"])
    # VIQ с NA-строкой всё равно должен посчитаться по оставшимся
    assert not pd.isna(s.men_mri_correlation["VIQ"])

def test_strongest_picks_max_abs_across_genders(tmp_path):
    rows = [
        ["Male",   90, 85, 80, 70, 170,  900000],
        ["Male",  100, 90, 85, 72, 175,  950000],
        ["Male",  110, 95, 90, 74, 180, 1000000],
        ["Female", 95, 90, 85, 60, 160,  920000],
        ["Female",100, 92, 88, 65, 165,  940000],
        ["Female",105, 94, 91, 70, 170,  960000],
    ]
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    best = max(
        max(abs(v) for v in s.men_mri_correlation.values() if not pd.isna(v)),
        max(abs(v) for v in s.women_mri_correlation.values() if not pd.isna(v)),
    )
    picked = max(
        abs(s.men_mri_correlation[s.strongest_mri_feature]),
        abs(s.women_mri_correlation[s.strongest_mri_feature]),
    )
    assert picked == pytest.approx(best, abs=1e-9)

def test_mri_count_not_in_features(tmp_path):
    rows = [
        ["Male", 90, 85, 80, 70, 170, 900000],
        ["Male", 110, 105, 100, 80, 180, 1000000],
    ]
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    assert "MRI_Count" not in s.men_mri_correlation
    assert "MRI_Count" not in s.women_mri_correlation

def test_single_row_per_gender_gives_all_nan(tmp_path):
    rows = [
        ["Male", 90, 85, 80, 70, 170, 900000],
        ["Female", 90, 85, 80, 60, 160, 900000],
    ]
    s = analyze_brain_correlations(write_brain(tmp_path, rows))
    assert all(pd.isna(v) for v in s.men_mri_correlation.values())
    assert all(pd.isna(v) for v in s.women_mri_correlation.values())