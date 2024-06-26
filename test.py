import json
import os

import numpy as np
import pandas as pd
import pytest

from some_folder.regular_use_functions import calculate_percentiles


# ABOUT THIS TEST:
# This test file is used to test the function calculate_percentiles from the file
# You need to generate the expected output for each test case using the notebook create_validation_ouputs.ipynb
# which is located in data/input/ref_baseline_saturation/ folder

# Initialize model settings
settings_country = {
    "Usa": SettingsUsa,
}

model_settings.init(
    country="Usa",
    user_settings={},
    country_settings=settings_country,
)

# Paths for data input and validation
DATA_INPUT_DIR = "./tests/unit/features/data/input/ref_baseline_saturation/"
VALIDATION_DIR = "./tests/unit/features/data/validation/ref_baseline_saturation/"

LIST_OF_SCENARIOS = [
    "features_df_np_missing",
    "features_df_ad_zerovalue",
    "features_df",
    "simple_features_start_w_zero",
    "simple_features_start_w_one",
]

LIST_OF_PERCENTILES = [
    -1,
    0,
    10,
    15,
    95,
    100,
    101,
]
# LIST_BASELINE = ["saturation"]
# LIST_BASELINE = ["baseline"]
LIST_BASELINE = ["saturation", "baseline"]


# Fixtures
@pytest.fixture
def create_parameters():
    def _create_parameters(df, indication_column, percentile):
        dicto = {}
        present_indications = df[indication_column].unique().tolist()
        for indication in present_indications:
            dicto[indication] = percentile
        return dicto

    return _create_parameters


@pytest.fixture
def get_df():
    def _get_df(dir, name):
        return pd.read_csv(os.path.join(dir, name), sep=";")

    return _get_df


@pytest.fixture
def get_json():
    def _get_json(dir, name):
        file_path = os.path.join(dir, name)
        with open(file_path, "r") as file:
            return json.load(file)

    return _get_json


# Test function
@pytest.mark.parametrize(
    "scenario,baseline,percentile",
    [
        (s, b, p)
        for s in LIST_OF_SCENARIOS
        for p in LIST_OF_PERCENTILES
        for b in LIST_BASELINE
    ],
)
def test_calculate_percentiles(
    scenario, baseline, percentile, create_parameters, get_df, get_json
):
    data = get_df(DATA_INPUT_DIR, f"{scenario}.csv")
    parameters = create_parameters(data, "sub_brand", percentile)
    output = calculate_percentiles(
        features_df=data.copy(),
        parameters=parameters,
        spend_col="spend_display_dtc",
        indication_geo_namescols=["sub_brand", "sub_national_code"],
        default_threshold=None,
        is_baseline=baseline == "baseline",
    )
    expected = get_json(VALIDATION_DIR, f"{scenario}_{baseline}_{percentile}.json")

    for key in expected.keys():
        for indication in expected[key].keys():
            np.testing.assert_almost_equal(
                output[key][indication], expected[key][indication], decimal=6
            )
