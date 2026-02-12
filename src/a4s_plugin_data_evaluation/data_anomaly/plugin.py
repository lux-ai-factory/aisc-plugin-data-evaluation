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


        measure_results["New Categories"] = self.__compute_new_categories(df_reference, df_evaluate, categorical_features)
        measure_results["Missing Categories"] = self.__compute_missing_categories(df_reference, df_evaluate, categorical_features)
        measure_results["Upper Constraint Violations"] = self.__compute_upper_constraint(df_evaluate, numerical_features, config.features)
        measure_results["Lower Constraint Violations"] = self.__compute_lower_constraint(df_evaluate, numerical_features, config.features)

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
                "name": "New Categories",
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
                "name": "Missing Categories",
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
                "name": "Upper Constraint Violations",
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
                "name": "Lower Constraint Violations",
                "score": float(min_violations),
                "description": f"{feature.name}",
            }
            return_measures.append(measure_lower_constraint)
        
        return return_measures

    @metric("New Categories")
    def new_categories(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get("New Categories", [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name="New Categories",
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results
    
    @metric("Missing Categories")
    def missing_categories(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get("Missing Categories", [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name="Missing Categories",
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results


    @metric("Upper Constraint Violations")
    def upper_constraint_violations(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get("Upper Constraint Violations", [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name="Upper Constraint Violations",
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results
    

    @metric("Lower Constraint Violations")
    def lower_constraint_violations(self, evaluate_result: dict) -> list[Measure]:
        list_measures = evaluate_result.get("measures_results", {}).get("Lower Constraint Violations", [])
        measure_results = []
        for measure_dict in list_measures:
            # Add measures
            measure_categories = Measure(
                name="Lower Constraint Violations",
                score=measure_dict.get("score"),
                description=measure_dict.get("description"),
            )
            measure_results.append(measure_categories)

        return measure_results


    @metric("Anomaly Pass")
    def aggregation_pass(self, evaluate_result: dict) -> list[Measure]:
        score_pass = 0
        for new_cat_measure in evaluate_result.get("measures_results", {}).get("New Categories", []):
            if new_cat_measure.get("score") == 0:
                score_pass += 1

        for missing_cat_measure in evaluate_result.get("measures_results", {}).get("Missing Categories", []):
            if missing_cat_measure.get("score") == 0: 
                score_pass += 1 


        for upper_constraint_measure in evaluate_result.get("measures_results", {}).get("Upper Constraint Violations", []):
            if upper_constraint_measure.get("score") == 0: 
                score_pass += 1

        for lower_constraint_measure in evaluate_result.get("measures_results", {}).get("Lower Constraint Violations", []):
            if lower_constraint_measure.get("score") == 0: 
                score_pass += 1

        measure_pass = Measure(name="Anomaly Pass", score=float(score_pass))        
        return [measure_pass]


    @metric("Anomaly Low")
    def aggregation_low(self, evaluate_result: dict) -> list[Measure]:
        score_low = 0
        for missing_cat_measure in evaluate_result.get("measures_results", {}).get("Missing Categories", []):
            if missing_cat_measure.get("score") > 0: 
                score_low += 1 

        measure_low = Measure(name="Anomaly Low", score=float(score_low))        

        return [measure_low]

    @metric("Anomaly Severe")
    def aggregation_severe(self, evaluate_result: dict) -> list[Measure]:
        score_severe = 0
        for missing_cat_measure in evaluate_result.get("measures_results", {}).get("New Categories", []):
            if missing_cat_measure.get("score") > 0: 
                score_severe += 1 

        for upper_constraint_measure in evaluate_result.get("measures_results", {}).get("Upper Constraint Violations", []):
            if upper_constraint_measure.get("score") > 0: 
                score_severe += 1
        
        for lower_constraint_measure in evaluate_result.get("measures_results", {}).get("Lower Constraint Violations", []):
            if lower_constraint_measure.get("score") > 0: 
                score_severe += 1

        measure_severe = Measure(name="Anomaly Severe", score=float(score_severe))        

        return [measure_severe]


    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        piechart = MetricVisualization(
            chart_type=ChartType.PIE, 
            metrics=[
                "Anomaly Pass",
                "Anomaly Low",
                "Anomaly Severe",
            ]
        )

        return [
            table, 
            piechart
        ]
