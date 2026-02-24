import pandas as pd

from paddy_yield_ml.pipelines import feature_prepare as fp


def test_parse_window_from_feature_patterns() -> None:
    assert fp.parse_window_from_feature("Min temp_D1_D30") == (1, 30)
    assert fp.parse_window_from_feature("30_50DRain( in mm)") == (30, 50)
    assert fp.parse_window_from_feature("30DRain( in mm)") == (1, 30)
    assert fp.parse_window_from_feature("NoWindowFeature") == (None, None)


def test_normalize_per_hectare_creates_target() -> None:
    df = pd.DataFrame(
        {
            "Hectares": [2, 4],
            "Paddy yield(in Kg)": [1000, 2400],
            "DAP_20days": [80, 160],
        }
    )

    out, created = fp.normalize_per_hectare(
        df=df,
        drop_original=False,
        create_input_scaled=True,
    )

    assert "Paddy yield_per_hectare(in Kg)" in out.columns
    assert out["Paddy yield_per_hectare(in Kg)"].tolist() == [500.0, 600.0]
    assert "DAP_20days_per_hectare" in out.columns
    assert "DAP_20days_per_hectare" in created


def test_feature_prepare_paths_point_to_clean_project_layout() -> None:
    assert fp.DATA_PATH.as_posix().endswith("data/input/paddydataset.csv")
    assert fp.DICT_PATH.as_posix().endswith("data/metadata/data_dictionary_paddy.csv")
    assert fp.OUT_DIR.as_posix().endswith("outputs/feature_prepare")


def test_screening_drops_dictionary_leakage_feature(tmp_path) -> None:
    prev_out = fp.OUT_DIR
    fp.OUT_DIR = tmp_path
    try:
        df = pd.DataFrame(
            {
                "Trash(in bundles)": [100, 110, 120],
                "Paddy yield(in Kg)": [1000, 1100, 1200],
                "Paddy yield_per_hectare(in Kg)": [500.0, 550.0, 600.0],
            }
        )
        role_map_df = pd.DataFrame(
            [
                {
                    "column_name": "Trash(in bundles)",
                    "final_role": "proxy",
                    "leakage_risk": "post_outcome_leakage",
                    "modeling_recommendation": "exclude_from_modeling",
                }
            ]
        )

        out = fp.build_feature_screening_summary(
            df=df,
            numeric_audit_df=pd.DataFrame(),
            categorical_overview_df=pd.DataFrame(),
            within_group_df=pd.DataFrame(),
            categorical_effect_df=pd.DataFrame(),
            proxy_df=pd.DataFrame(),
            corr_pairs_df=pd.DataFrame(),
            role_map_df=role_map_df,
        )
        row = out[out["feature"] == "Trash(in bundles)"].iloc[0]
        assert row["recommendation"] == "drop"
        assert "dictionary_proxy_or_leakage" in str(row["reasons"])
    finally:
        fp.OUT_DIR = prev_out
