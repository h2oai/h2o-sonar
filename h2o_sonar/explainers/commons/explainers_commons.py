# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import abc
import json
import os

import datatable

from h2o_sonar.lib.api import datasets
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import persistences


class AbstractFeatureImportanceExplainer(abc.ABC):
    """Feature importance explainers commons - shared parent of all feature
    importance explainers ensures that all of them will create identical
    (normalized) explanations. This ensures that user can just choose the method (naive,
    Kernel SHAP, permutation-based, ...) and always get exactly the same format of
    explanations.

    Feature importance explainers might be easily customized by copying methods from
    this abstract class to the explainer so that the changes / customization might be
    done there (parent implementation override).

    """

    PARAM_SAMPLE = "sample"
    PARAM_SAMPLE_SIZE = "sample_size"
    PARAM_FAST_APPROX = "fast_approx_contrib"
    PARAM_CALCULATE_PREDICTIONS = "calculate_predictions"

    # do NOT convert (e.g. Pandas to datatable) big dataset to avoid OOM (2x dataset)
    DATASET_SAMPLING_NATIVE = int(1e6)
    DATASET_SAMPLING_SEED = 972_458
    H2O_FRAME_INDEX = "h2oframe_idx"

    # file system constants
    LOCAL_BASE_PATH = "local_base_path"
    GLOBAL_BASE_PATH = "global_base_path"
    LOCAL_FILE_MAPPING = "local_file_mapping"
    GLOBAL_FILE_MAPPING = "global_file_mapping"

    def __init__(self):
        self.model = None
        self.model_meta = None
        self.persistence = None
        self.sanitization_map = None
        self.config = None
        self.logger = None

    def setup(
        self,
        model: models.ExplainableModel,
        persistence: persistences.ExplainerPersistence,
        logger,
    ):
        self.model = model
        self.model_meta = model.meta
        self.persistence = persistence
        self.sanitization_map = model.meta.sanitization_map
        self.logger = logger

    def _normalize_frames_to_gom(
        self,
        explanation: e10s.GlobalFeatImpExplanation,
        shapley_means_dict: dict[str, datatable.Frame],
        file_mapping_dict: dict[str, str],
    ) -> f5s.GlobalFeatImpJSonDatatableFormat:
        json_representation = f5s.GlobalFeatImpJSonDatatableFormat(
            explanation=explanation,
            json_data=json.dumps(file_mapping_dict, indent=4),
        )

        for label, shapley_means in shapley_means_dict.items():
            sanitized_label = self.sanitization_map.sanitize_value(str(label))
            gom_data_frame: datatable.Frame = datatable.Frame(
                {
                    f5s.GlobalFeatImpJSonDatatableFormat.COL_NAME: [],
                    f5s.GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE: [],
                    f5s.GlobalFeatImpJSonDatatableFormat.COL_GLOBAL_SCOPE: [],
                }
            )
            for name in shapley_means.names:
                gom_data_frame.rbind(
                    datatable.Frame(
                        {
                            f5s.GlobalFeatImpJSonDatatableFormat.COL_NAME: [name],
                            f5s.GlobalFeatImpJSonDatatableFormat.COL_IMPORTANCE: [
                                shapley_means[0, name]
                            ],
                            f5s.GlobalFeatImpJSonDatatableFormat.COL_GLOBAL_SCOPE: [
                                True
                            ],
                        }
                    )
                )
            total_rows = gom_data_frame.shape[0]
            file_name: str = file_mapping_dict[
                f5s.GlobalFeatImpJSonDatatableFormat.KEY_FILES
            ][sanitized_label]
            json_representation.add_data_frame(
                format_data=gom_data_frame, file_name=file_name
            )
            json_representation.update_index_file(
                file_mapping_dict, total_rows=total_rows
            )

        return json_representation

    def _normalize_local_shapley_frames_to_gom(
        self,
        local_explanation: e10s.LocalFeatImpExplanation,
        shapley_contribs_dict: dict[str, datatable.Frame],
        file_mapping_dict: dict,
    ) -> f5s.LocalFeatImpWithYhatsJSonDatatableFormat:
        # this explainer should provide local explanations by itself
        file_mapping_dict[
            f5s.LocalFeatImpWithYhatsJSonDatatableFormat.KEY_ON_DEMAND
        ] = True

        json_representation = f5s.LocalFeatImpWithYhatsJSonDatatableFormat(
            explanation=local_explanation,
            json_data=json.dumps(file_mapping_dict, indent=4),
        )

        for label, shapley_contribs in shapley_contribs_dict.items():
            sanitized_label = self.model_meta.sanitization_map.sanitize_value(
                str(label)
            )
            file_name: str = file_mapping_dict[
                f5s.LocalFeatImpWithYhatsJSonDatatableFormat.KEY_FILES
            ][sanitized_label]
            json_representation.add_data_frame(
                format_data=shapley_contribs, file_name=file_name
            )
            json_representation.update_index_file(
                file_mapping_dict, total_rows=shapley_contribs.shape[0]
            )

        return json_representation

    def _export_shapley_frames(
        self,
        shapley_df: datatable.Frame,
        target_dir: str,
        export_original_features: bool,
    ):
        this_class = AbstractFeatureImportanceExplainer

        if export_original_features:
            shapley_bin_name = "shapley.orig.feat.bin"
            shapley_csv_name = "shapley.orig.feat.csv"
            shapley_formatted_name = "shapley_formatted_orig_feat.csv"
        else:
            shapley_bin_name = "shapley.bin"
            shapley_csv_name = "shapley.csv"
            shapley_formatted_name = "shapley_formatted.csv"

        # filter down to features with importance > 0
        shapley_df = datasets.filter_importance_greater_than_zero(
            frame=shapley_df, skip_bias=False
        )

        # write out Shapley frames
        datasets.DatasetApi.write_dataset(
            shapley_df, os.path.join(target_dir, shapley_bin_name)
        )
        datasets.DatasetApi.write_csv(
            dataset=shapley_df, path=os.path.join(target_dir, shapley_csv_name)
        )

        # formatted Shapley for transformed features only for now
        shapley_df_formatted = self._format_rc_shapley(
            raw=shapley_df,
            format_bias=False,
        )
        if this_class.H2O_FRAME_INDEX in shapley_df.names:
            if shapley_df.shape[0] == shapley_df_formatted.shape[0]:
                shapley_df_formatted[:, this_class.H2O_FRAME_INDEX] = shapley_df[
                    :, this_class.H2O_FRAME_INDEX
                ]
            else:
                self.logger.info(
                    f"Shapley frame and formatted Shapley frame "
                    f"have different number of rows! Shapley frame "
                    f"rows: {shapley_df.shape[0]}, Shapley "
                    f"formatted frame rows: "
                    f"{shapley_df_formatted.shape[0]}",
                )
        # write out formatted Shapley
        csv_file_path = os.path.join(target_dir, shapley_formatted_name)
        datasets.DatasetApi.write_csv(dataset=shapley_df_formatted, path=csv_file_path)
        datasets.DatasetApi.zip_csv(csv_file_path)
        # remove CSV as it's no longer needed
        os.remove(csv_file_path)

        del shapley_df

    @staticmethod
    def _export_shapley_frames_multinomial(
        shapley_df: datatable.Frame,
        target_dir: str,
        export_original_features: bool,
        label,
        skip_bias: bool = True,
    ):
        if export_original_features:
            shapley_bin_name = "shapley.multiclass.orig.feat.bin"
            shapley_csv_name = "shapley.multiclass.orig.feat.csv"
        else:
            shapley_bin_name = "shapley.multiclass.bin"
            shapley_csv_name = "shapley.multiclass.csv"

        # filter down to features with importance > 0
        shapley_df = datasets.filter_importance_greater_than_zero(
            frame=shapley_df,
            label=label,
            skip_bias=skip_bias,
        )

        # write out data for transformed features
        datasets.DatasetApi.write_dataset(
            shapley_df, os.path.join(target_dir, shapley_bin_name)
        )
        csv_file_path = os.path.join(target_dir, shapley_csv_name)
        datasets.DatasetApi.write_csv(shapley_df, csv_file_path)
        datasets.DatasetApi.zip_csv(csv_file_path)
        # remove CSV as it's no longer needed
        os.remove(csv_file_path)

    @staticmethod
    def _filter_importance_greater_than_zero(frame, label=None):
        feature_list = list(frame.names)
        bias_var = (
            datasets.ExplainableDataset.COL_BIAS
            if label is None or datasets.ExplainableDataset.COL_BIAS in feature_list
            else f"{datasets.ExplainableDataset.COL_BIAS}.{label}"
        )
        if bias_var not in feature_list:
            raise RuntimeError(
                f"Label '{bias_var}' not in Shapley frame with columns: {feature_list}"
            )
        feature_list.remove(bias_var)
        summed_array = frame[:, feature_list].sum().to_numpy()[0]

        non_zero_importance = [
            col_name for i, col_name in enumerate(feature_list) if summed_array[i] != 0
        ]

        # only drop all features with 0 importance - if NOT all features have 0
        # importance as is the case for constant models
        if summed_array.any():
            non_zero_importance.append(bias_var)
            frame = frame[:, non_zero_importance]

        return frame

    def _format_rc_shapley(self, raw, format_bias: bool = True, **kwargs):
        """Format Shapley reason codes into format:

        rc_1_var_name | rc_1_contrib | rc_2_var_name | rc_2_contrib | ... |
        rc_p_var_name | rc_p_contrib

        Parameters
        ----------
        X : str or dt.Frame
          Original Shapley reason code file downloaded from MLI GUI.
        format_bias : bool
          Do format bias column.

        Returns
        -------
        dt.Frame :
           Reformatted reason code frame.

        """
        self.logger.debug("Normalizing format of Shapley reason codes...")
        names = ["name_" + str(i) for i in range(1, len(raw.names))]
        contribs = ["contrib_" + str(i) for i in range(1, len(raw.names))]
        columns_ = [
            elem for pair in zip(names, contribs, strict=False) for elem in pair
        ]
        # for Shapley bias should be the same
        if format_bias:
            bias = raw[0, datasets.ExplainableDataset.COL_BIAS]
        else:
            bias = None

        rows_in_cache = 0
        last_read_from = 0
        last_read_to = 0
        pandas_cache = None

        data = []
        for row in range(raw.nrows):
            if rows_in_cache == 0:
                remaining_rows = raw.nrows - last_read_to
                rows = remaining_rows if remaining_rows < 10000 else 10000
                pandas_cache = raw[last_read_to : last_read_to + rows, :].to_pandas()
                rows_in_cache = rows
                last_read_from = last_read_to
                last_read_to += rows

            # TODO fix this sorting with datatable + delete result of  to_pandas()?
            sorted_row = pandas_cache.iloc[row - last_read_from, :-1].sort_values(
                axis=0, ascending=False
            )
            contribs = list(sorted_row)
            names = list(sorted_row.index)
            vals = [
                elem for pair in zip(names, contribs, strict=False) for elem in pair
            ]
            data.append(dict(zip(columns_, vals, strict=False)))
            rows_in_cache -= 1

        formatted_frame = datatable.Frame(data, names=columns_)
        if format_bias:
            formatted_frame[:, datasets.ExplainableDataset.COL_BIAS] = bias

        job_id = kwargs.get("job_id")
        if job_id is not None:
            # trick to make it work for parallelize work
            return job_id, formatted_frame
        return formatted_frame
