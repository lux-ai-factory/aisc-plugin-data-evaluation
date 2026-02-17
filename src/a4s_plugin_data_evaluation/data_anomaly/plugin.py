from enum import Enum
import io
import logging
from logging import config
from typing import Any
import zipfile
import pandas as pd

from a4s_plugin_interface import metric
from a4s_plugin_interface import BaseEvaluationPlugin, PluginFeatureFlags
from a4s_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType
from pydantic import BaseModel, Field, model_validator

from ..utils import ConfigForm, Feature, FeatureType, BaseDataPlugin


NEW_CATEGORIES_MEASURE_NAME = "New Categories"
MISSING_CATEGORIES_MEASURE_NAME = "Missing Categories"
UPPER_CONSTRAINT_VIOLATIONS_MEASURE_NAME = "Upper Constraint Violations"
LOWER_CONSTRAINT_VIOLATIONS_MEASURE_NAME = "Lower Constraint Violations"
DISTRIBUTION_OUTLIER_MEASURE_NAME = "Distribution Outlier"
ANOMALY_PASS_MEASURE_NAME = "Anomaly Pass"
ANOMALY_LOW_MEASURE_NAME = "Anomaly Low"
ANOMALY_SEVERE_MEASURE_NAME = "Anomaly Severe"


logger = logging.getLogger(__name__)

class ZipDatasetInputProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes) -> Any:
        import pandas as pd

        fnames = ("train.csv", "test.csv")

        files = {}
        with zipfile.ZipFile(io.BytesIO(file_content)) as z:
            for name in z.namelist():
                if name in fnames:
                    files[name] = z.read(name)

        assert len(files) == 2

        res = {
            "reference": pd.read_csv(io.BytesIO(files[fnames[0]])),
            "evaluate": pd.read_csv(io.BytesIO(files[fnames[1]])),
        }

        return res


class DataAnomalyPlugin(BaseDataPlugin):
    def parse_config_from_dataset(self) -> dict | None:
        config: ConfigForm = ConfigForm(
            features=[],
            date_feature=None,
            target_feature=None,
        )

        df: pd.DataFrame = self.get_dataset().get("reference", pd.DataFrame())

        for col_name in df.columns:
            col_data = df[col_name]

            feature_type = FeatureType.CATEGORICAL

            # Check for Date
            if pd.api.types.is_datetime64_any_dtype(col_data):
                feature_type = FeatureType.DATE
            elif pd.api.types.is_object_dtype(col_data):
                temp = pd.to_datetime(col_data, errors="coerce")
                if temp.isna().any():
                    logger.warning(
                        f"Attempted to parse {col_name} as a date, but failed."
                    )
                else:
                    feature_type = FeatureType.DATE

            # Check for Numeric
            if feature_type != FeatureType.DATE:
                if pd.api.types.is_integer_dtype(col_data):
                    feature_type = FeatureType.INTEGER
                elif pd.api.types.is_float_dtype(col_data):
                    feature_type = FeatureType.FLOAT

            # Get Min/Max for Numeric types
            if feature_type in [FeatureType.INTEGER, FeatureType.FLOAT]:
                col_min = float(col_data.min()) if not pd.isna(col_data.min()) else 0.0
                col_max = float(col_data.max()) if not pd.isna(col_data.max()) else 0.0
            else:
                # For Categorical or Date, min/max usually aren't numeric ranges
                col_min = 0.0
                col_max = 0.0

            feature: Feature = Feature(
                name=col_name, min=col_min, max=col_max, type=feature_type
            )
            config.features.append(feature)

        return config.model_dump()


    def set_dataset_input_provider(
        self, file_content: bytes | None
    ) -> BaseInputProvider:
        self.dataset_input_provider = ZipDatasetInputProvider(file_content)
        return self.dataset_input_provider


    @property
    def display_icon(self) -> str:
        return "flag"

    def evaluate(self, config_data: dict) -> list[Measure]:
        import pandas as pd

        measure_results = dict()
        config = self.validate_config_form_data(config_data)

        categorical_features = [
            f.name for f in config.features if f.type is FeatureType.CATEGORICAL
        ]
        numerical_features = [
            f.name for f in config.features if f.type is FeatureType.INTEGER or f.type is FeatureType.FLOAT
        ]


        df_evaluate: pd.DataFrame = self.get_dataset().get("evaluate", pd.DataFrame())
        df_reference: pd.DataFrame = self.get_dataset().get("reference", pd.DataFrame())

        # TODO: Should be reworked
        measure_results[NEW_CATEGORIES_MEASURE_NAME] = self.__compute_new_categories(df_reference, df_evaluate, categorical_features)
        measure_results[MISSING_CATEGORIES_MEASURE_NAME] = self.__compute_missing_categories(df_reference, df_evaluate, categorical_features)
        measure_results[UPPER_CONSTRAINT_VIOLATIONS_MEASURE_NAME] = self.__compute_upper_constraint(df_evaluate, numerical_features, config.features)
        measure_results[LOWER_CONSTRAINT_VIOLATIONS_MEASURE_NAME] = self.__compute_lower_constraint(df_evaluate, numerical_features, config.features)
        measure_results[DISTRIBUTION_OUTLIER_MEASURE_NAME] = self.__compute_distribution(df_reference, df_evaluate, numerical_features)

        return {
            "measures_results": measure_results
        }

    def __compute_new_categories(self, 
            df_reference: pd.DataFrame, 
            df_evaluate: pd.DataFrame, 
            categorical_features: list[str]
    ) -> list[dict[str, Any]]:
        return_measures = []
        for feat_name in categorical_features:
            # Categories
            ref_categories = set(df_reference[feat_name].unique())
            eval_categories = set(df_evaluate[feat_name].unique())

            new_categories = eval_categories - ref_categories  # set difference

            # Add measures
            measure_categories = {
                "score": float(len(new_categories)),
                "description": f"{feat_name}",
            }
            return_measures.append(measure_categories)
        
        return return_measures
    
    def __compute_missing_categories(self, 
            df_reference: pd.DataFrame, 
            df_evaluate: pd.DataFrame, 
            categorical_features: list[str]
    ) -> list[dict[str, Any]]:
        return_measures = []
        for feat_name in categorical_features:
            # Categories
            ref_categories = set(df_reference[feat_name].unique())
            eval_categories = set(df_evaluate[feat_name].unique())

            missing_categories = ref_categories - eval_categories  # set difference

            # Add measures
            measure_categories = {
                "score": float(len(missing_categories)),
                "description": f"{feat_name}",
            }
            return_measures.append(measure_categories)
        
        return return_measures
    

    def __compute_upper_constraint(self, 
            df_evaluate: pd.DataFrame, 
            numerical_features: list[str],
            config_features: list[Feature]
    ) -> list[dict[str, Any]]:
        return_measures = []
        for feature in config_features:
            if feature.name not in numerical_features: 
                continue

            values = df_evaluate[feature.name]
            max_violations = (values > feature.max).sum()

            # Add measures
            measure_upper_constraint = {
                "score": float(max_violations),
                "description": f"{feature.name}",
            }
            return_measures.append(measure_upper_constraint)
        
        return return_measures
    
    def __compute_lower_constraint(self, 
            df_evaluate: pd.DataFrame, 
            numerical_features: list[str],
            config_features: list[Feature]
    ) -> list[dict[str, Any]]:
        return_measures = []
        for feature in config_features:
            if feature.name not in numerical_features: 
                continue

            values = df_evaluate[feature.name]
            min_violations = (values < feature.min).sum()

            # Add measures
            measure_lower_constraint = {
                "score": float(min_violations),
                "description": f"{feature.name}",
            }
            return_measures.append(measure_lower_constraint)
        
        return return_measures


    def __compute_distribution(self, 
            df_reference: pd.DataFrame, 
            df_evaluate: pd.DataFrame, 
            numerical_features: list[str]
    ) -> list[dict[str, Any]]:
        return_measures = []
        for feat_name in numerical_features:
            ref_mean = df_reference[feat_name].mean()
            ref_std = df_reference[feat_name].std()

            values = df_evaluate[feat_name]
            
            dist_violations = ((values < ref_mean - 3 * ref_std) | (values > ref_mean + 3 * ref_std)).sum()

            # Add measures
            measure_distribution = {
                "score": float(dist_violations),
                "description": f"{feat_name}",
            }
            return_measures.append(measure_distribution)
        
        return return_measures

    @metric(NEW_CATEGORIES_MEASURE_NAME)
    def new_categories(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get(NEW_CATEGORIES_MEASURE_NAME, [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name=NEW_CATEGORIES_MEASURE_NAME,
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results
    
    @metric(MISSING_CATEGORIES_MEASURE_NAME)
    def missing_categories(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get(MISSING_CATEGORIES_MEASURE_NAME, [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name=MISSING_CATEGORIES_MEASURE_NAME,
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results


    @metric(UPPER_CONSTRAINT_VIOLATIONS_MEASURE_NAME)
    def upper_constraint_violations(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get(UPPER_CONSTRAINT_VIOLATIONS_MEASURE_NAME, [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name=UPPER_CONSTRAINT_VIOLATIONS_MEASURE_NAME,
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results
    

    @metric(LOWER_CONSTRAINT_VIOLATIONS_MEASURE_NAME)
    def lower_constraint_violations(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get(LOWER_CONSTRAINT_VIOLATIONS_MEASURE_NAME, [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name=LOWER_CONSTRAINT_VIOLATIONS_MEASURE_NAME,
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results
    

    @metric(DISTRIBUTION_OUTLIER_MEASURE_NAME)
    def distribution_outlier(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get(DISTRIBUTION_OUTLIER_MEASURE_NAME, [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name=DISTRIBUTION_OUTLIER_MEASURE_NAME,
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results


    @metric(ANOMALY_PASS_MEASURE_NAME)
    def aggregation_pass(self, evaluate_result: dict) -> list[Measure]:
        score_pass = 0
        for new_cat_measure in evaluate_result.get("measures_results", {}).get(NEW_CATEGORIES_MEASURE_NAME, []):
            if new_cat_measure.get("score") == 0:
                score_pass += 1

        for missing_cat_measure in evaluate_result.get("measures_results", {}).get(MISSING_CATEGORIES_MEASURE_NAME, []):
            if missing_cat_measure.get("score") == 0: 
                score_pass += 1 

        for upper_constraint_measure in evaluate_result.get("measures_results", {}).get(UPPER_CONSTRAINT_VIOLATIONS_MEASURE_NAME, []):
            if upper_constraint_measure.get("score") == 0: 
                score_pass += 1

        for lower_constraint_measure in evaluate_result.get("measures_results", {}).get(LOWER_CONSTRAINT_VIOLATIONS_MEASURE_NAME, []):
            if lower_constraint_measure.get("score") == 0: 
                score_pass += 1
        
        for distribution_measure in evaluate_result.get("measures_results", {}).get(DISTRIBUTION_OUTLIER_MEASURE_NAME, []):
            if distribution_measure.get("score") == 0:
                score_pass += 1

        measure_pass = Measure(name=ANOMALY_PASS_MEASURE_NAME, score=float(score_pass))        
        return [measure_pass]


    @metric(ANOMALY_LOW_MEASURE_NAME)
    def aggregation_low(self, evaluate_result: dict) -> list[Measure]:
        score_low = 0
        for missing_cat_measure in evaluate_result.get("measures_results", {}).get(MISSING_CATEGORIES_MEASURE_NAME, []):
            if missing_cat_measure.get("score") > 0: 
                score_low += 1 
        
        for distribution_measure in evaluate_result.get("measures_results", {}).get(DISTRIBUTION_OUTLIER_MEASURE_NAME, []): 
            if distribution_measure.get("score") > 0: 
                score_low += 1

        measure_low = Measure(name=ANOMALY_LOW_MEASURE_NAME, score=float(score_low))        

        return [measure_low]

    @metric(ANOMALY_SEVERE_MEASURE_NAME)
    def aggregation_severe(self, evaluate_result: dict) -> list[Measure]:
        score_severe = 0
        for missing_cat_measure in evaluate_result.get("measures_results", {}).get(NEW_CATEGORIES_MEASURE_NAME, []):
            if missing_cat_measure.get("score") > 0: 
                score_severe += 1 

        for upper_constraint_measure in evaluate_result.get("measures_results", {}).get(UPPER_CONSTRAINT_VIOLATIONS_MEASURE_NAME, []):
            if upper_constraint_measure.get("score") > 0: 
                score_severe += 1
        
        for lower_constraint_measure in evaluate_result.get("measures_results", {}).get(LOWER_CONSTRAINT_VIOLATIONS_MEASURE_NAME, []):
            if lower_constraint_measure.get("score") > 0: 
                score_severe += 1

        measure_severe = Measure(name=ANOMALY_SEVERE_MEASURE_NAME, score=float(score_severe))        

        return [measure_severe]


    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        piechart = MetricVisualization(
            chart_type=ChartType.PIE, 
            metrics=[
                ANOMALY_PASS_MEASURE_NAME,
                ANOMALY_LOW_MEASURE_NAME,
                ANOMALY_SEVERE_MEASURE_NAME,
            ]
        )

        return [
            table, 
            piechart
        ]
