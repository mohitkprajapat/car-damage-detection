import os
import pandas as pd
from src import config
from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.metrics import *
from evidently.presets import *
from evidently.tests import *

from evidently.future.tests import Reference

REFERENCE_PATH = os.path.join(config.root_dir, "monitoring", "reference_data.csv")
PRODUCTION_LOG_PATH = os.path.join(config.root_dir, "monitoring", "production_data.csv")
HTML_PATH = os.path.join(config.root_dir, "static", "monitor_report.html")


def evi_monitor():
    cat_column = ["true_label","pred_class"]
    num_column = ["prob_minor","prob_moderate","prob_severe","confidence","damage_score"]
    drift_column = ["prob_minor", "prob_moderate", "prob_severe", "confidence", "damage_score", "pred_class"]

    df_ref = pd.read_csv(REFERENCE_PATH)
    df_prod = pd.read_csv(PRODUCTION_LOG_PATH)
    schema = DataDefinition(
        numerical_columns=num_column,
        categorical_columns=cat_column,

        classification=[MulticlassClassification(
            target="true_label",
            prediction_labels="pred_class",
            prediction_probs=["prob_minor", "prob_moderate", "prob_severe"],
        )]
    )

    eval_ref = Dataset.from_pandas(
        pd.DataFrame(df_ref),
        data_definition=schema
    )

    eval_prod = Dataset.from_pandas(
        pd.DataFrame(df_prod),
        data_definition=schema
    )

    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(columns=drift_column),
            ClassificationPreset()
        ],
    include_tests=True)
    my_eval = report.run(eval_prod, eval_ref)
    my_eval.save_html(HTML_PATH)