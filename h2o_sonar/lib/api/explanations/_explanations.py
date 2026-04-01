# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import traceback

import airium

from h2o_sonar import loggers
from h2o_sonar.lib.api import commons
from h2o_sonar.lib.api import formats as f5s
from h2o_sonar.lib.api.explanations import _explanations_base


class GlobalDataFrameExplanation(_explanations_base.Explanation):
    """Generic explanation which doesn't fit any other type."""

    _explanation_type = "frame"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalDataFrameExplanation(_explanations_base.Explanation):
    """Generic explanation which doesn't fit any other type."""

    _explanation_type = "frame"
    _is_global = False

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class PartialDependenceExplanation(_explanations_base.Explanation):
    _explanation_type = "partial-dependence"
    _is_global = True

    # keyword which indicates explainer can add PD for feature using global explain
    KEYWORD_CAN_ADD_FEATURE = "can-add-feature"

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class IndividualConditionalExplanation(_explanations_base.Explanation):
    _explanation_type = "individual-conditional-explanation"
    _is_global = False

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalRuleExplanation(_explanations_base.Explanation):
    _explanation_type = "rule"
    _is_global = False

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalRuleExplanation(_explanations_base.Explanation):
    _explanation_type = "rule"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalFeatImpExplanation(_explanations_base.Explanation):
    _explanation_type = "feature-importance"
    _is_global = False
    _format_types = [
        commons.MimeType.MIME_DATATABLE,
        commons.MimeType.MIME_CSV,
        commons.MimeType.MIME_JSON,
    ]

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalNlpLocoExplanation(_explanations_base.Explanation):
    _explanation_type = "nlp-loco"
    _is_global = False
    _format_types = [commons.MimeType.MIME_JSON]

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalHtmlSnippetExplanation(_explanations_base.Explanation):
    _explanation_type = "html-snippet"
    _is_global = False
    _format_types = [
        commons.MimeType.MIME_HTML,
    ]

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalTextSnippetExplanation(_explanations_base.Explanation):
    _explanation_type = "text-snippet"
    _is_global = False
    _format_types = [
        commons.MimeType.MIME_TEXT,
    ]

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalSummaryFeatImpExplanation(_explanations_base.Explanation):
    _explanation_type = "summary-feature-importance"
    _is_global = False
    _format_types = [
        commons.MimeType.MIME_JSON,
    ]

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalFeatImpExplanation(_explanations_base.Explanation):
    _explanation_type = "feature-importance"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalGroupedBarChartExplanation(_explanations_base.Explanation):
    _explanation_type = "grouped-bar-chart"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalNlpLocoExplanation(_explanations_base.Explanation):
    _explanation_type = "nlp-loco"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalSummaryFeatImpExplanation(_explanations_base.Explanation):
    _explanation_type = "summary-feature-importance"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalScatterPlotExplanation(_explanations_base.Explanation):
    _explanation_type = "scatter-plot"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalLinePlotExplanation(_explanations_base.Explanation):
    _explanation_type = "line-plot"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalDtExplanation(_explanations_base.Explanation):
    _explanation_type = "decision-tree"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocalDtExplanation(_explanations_base.Explanation):
    _explanation_type = "decision-tree"
    _is_global = False

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class TextExplanation(_explanations_base.Explanation):
    _explanation_type = "text-explanation"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class DiaExplanation(_explanations_base.Explanation):
    _explanation_type = "disparate-impact-analysis"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class SaExplanation(_explanations_base.Explanation):
    _explanation_type = "sensitivity-analysis"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class TimeSeriesAppExplanation(_explanations_base.Explanation):
    _explanation_type = "time-series-app"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class LocoExplanation(_explanations_base.Explanation):
    _explanation_type = "loco"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class NlpTokenizerExplanation(_explanations_base.Explanation):
    _explanation_type = "nlp-tokenizer"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class OnDemandExplanation(_explanations_base.Explanation):
    """On-demand explanations typically used for ad-hoc local on-demand
    explainer execution by the explainer executor.

    """

    _explanation_type = "on-demand"
    _is_global = False

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class ReportExplanation(_explanations_base.Explanation):
    """Generic report explanation provides various document formats (like Word,
    Markdown, ...) explanations.

    """

    _explanation_type = "report"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class GlobalHtmlFragmentExplanation(_explanations_base.Explanation):
    _explanation_type = "html-fragment"
    _is_global = True
    _format_types = [
        commons.MimeType.MIME_HTML,
    ]

    def __init__(
        self,
        explainer=None,
        evaluator=None,
        display_name: str = None,
        display_category: str = None,
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer or evaluator,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None

    def add_html_format(self, html: str):
        """Add HTML format."""
        self.add_format(
            f5s.HtmlFormat(
                explanation=self,
                format_data=html,
                persistence=self.explainer.persistence.store,
            )
        )

    @staticmethod
    def from_explanation(
        explainer,
        explanation,
        display_name: str = None,
        display_category: str = None,
        absolute_paths: bool = False,
        problems: dict = None,
        is_raw_feature: bool = True,
        data_as_text: bool = True,
        logger=None,
    ) -> "GlobalHtmlFragmentExplanation":
        """Create HTML fragment explanation:

        - from ``GlobalFeatImpExplanation``
           - with ``formats.HtmlFormat``
        - from ``PartialDependenceJSonFormat``
           - with ``formats.HtmlFormat``

        Parameters
        ----------
        explainer :
          Explainer instance.
        explanation :
          Explanation instance.
        display_name : str
          Custom display name.
        display_category : str
          Custom display category.
        absolute_paths : bool
          ``True`` to create HTML representation with absolute paths to images and
          explanations, else ``False`` to create relative paths (default).
        problems : dict
          Dictionary of class to feature names with features which are problematic
          to highlight their charts.
        is_raw_feature : bool
          ``True`` if input explains original features, else ``False`` for
          transformed features.
        data_as_text : bool
          Generate HTML text for the chart data.
        logger :
          Optional logger.

        """
        problems = problems or {}
        logger = logger or loggers.SonarPrintLogger()

        html_explanation = GlobalHtmlFragmentExplanation(
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )
        t_html_explanation = html_explanation.explanation_type()
        html_format = f5s.HtmlFormat(
            explanation=html_explanation,
            format_data=f5s.HtmlFormat.MINIMAL_HTML,
            persistence=explainer.persistence.store,
        )

        html_src = airium.Airium()
        result = explainer.get_result()
        persistence = explainer.persistence

        if isinstance(explanation, GlobalFeatImpExplanation):
            idx_dict: dict = f5s.GlobalFeatImpJSonFormat.load_index_file(
                persistence=persistence,
                explanation_type=GlobalFeatImpExplanation.explanation_type(),
            )
            classes = set()
            if idx_dict:
                idx_files = idx_dict.get(f5s.GlobalDtJSonFormat.KEY_FILES, None)
                if idx_files:
                    classes = list(idx_files.keys())
            classes = classes or [None]

            for i, clazz in enumerate(classes):
                with html_src.b():
                    msg_suffix = f" for the class '{clazz}'" if clazz else ""
                    html_src(f"Feature importance{msg_suffix}:")
                with html_src.div():
                    file_path = persistence.get_explanation_file_path(
                        explanation_type=t_html_explanation,
                        explanation_format=html_format.mime,
                        explanation_file=f"fi-class-{i}.png",
                    )
                    result.plot(clazz=clazz, file_path=file_path)
                    html_src.img(
                        src=(
                            persistence.get_relative_path(file_path)
                            if not absolute_paths
                            else file_path
                        ),
                        alt=f"Feature importance for class '{clazz}'",
                        # ensure that image will not overflow enclosing <div/>
                        style=(
                            "height: 100%; max-width: 100%; display: block; "
                            "margin: auto;"
                        ),
                    )
                    html_src.br()
                    if data_as_text:
                        try:
                            data = result.data()
                            feature = data[0, :].to_dict().get("feature", [""])[0]
                            with html_src.p():
                                html_src("The most important ")
                                if is_raw_feature:
                                    html_src("original")
                                else:
                                    html_src("transformed")
                                if clazz == f5s.ExplanationFormat.LABEL_REGRESSION:
                                    html_src(" feature")
                                else:
                                    html_src(" feature of the class ")
                                    with html_src.code():
                                        html_src(f"{clazz}")
                                html_src(" is ")
                                with html_src.code():
                                    html_src(f"{feature}")
                                html_src(".")
                            with html_src.p():
                                if is_raw_feature:
                                    html_src("Original")
                                else:
                                    html_src("Transformed")
                                if clazz == f5s.ExplanationFormat.LABEL_REGRESSION:
                                    html_src(" feature importances:")
                                else:
                                    html_src(
                                        f" feature importances for the class '{clazz}':"
                                    )
                                with html_src.ul():
                                    for ii in range(data.shape[0]):
                                        row_dict = data[ii, :].to_dict()
                                        feature = row_dict.get("feature", [""])[0]
                                        importance = row_dict.get("importance", [0])[0]
                                        with html_src.li():
                                            html_src(f"{ii + 1}. ")
                                            with html_src.code():
                                                html_src(f"{feature}")
                                            html_src(" feature with importance ")
                                            with html_src.code():
                                                html_src(f"{importance}")
                        except Exception as ex:
                            logger.warning(
                                f"Unable to create HTML representation feature "
                                f"importance text for class '{clazz}' due to: {ex}\n"
                                f"{traceback.format_exc()}"
                            )

            html_format.update_data(
                str(html_src), f"{persistence.FILE_EXPLANATION}.html"
            )

            html_explanation.add_format(html_format)

            return html_explanation

        elif isinstance(explanation, PartialDependenceExplanation):
            idx_dict: dict = f5s.PartialDependenceJSonFormat.load_index_file(
                persistence=persistence,
                explanation_type=PartialDependenceExplanation.explanation_type(),
            )
            features = []
            classes = set()
            if idx_dict:
                idx_features = idx_dict.get(
                    f5s.PartialDependenceJSonFormat.KEY_FEATURES, None
                )
                if idx_features:
                    for i_f, feature in enumerate(idx_features):
                        idx_feature = idx_dict[
                            f5s.PartialDependenceJSonFormat.KEY_FEATURES
                        ][feature]
                        features.append(idx_feature)
                        idx_classes = idx_feature.get(
                            f5s.PartialDependenceJSonFormat.KEY_FILES, None
                        )
                        if idx_classes:
                            for i_c, clazz in enumerate(idx_classes.keys()):
                                classes.add(clazz)

                                with html_src.b():
                                    msg_suffix = (
                                        f" for the feature '{feature}'"
                                        f" and class '{clazz}'"
                                    )
                                    html_src(f"Partial Dependence Plot{msg_suffix}:")
                                with html_src.div():
                                    file_path = persistence.get_explanation_file_path(
                                        explanation_type=t_html_explanation,
                                        explanation_format=html_format.mime,
                                        explanation_file=(
                                            f"pd-feature-{i_f}-class-{i_c}.png"
                                        ),
                                    )
                                    result.plot(
                                        feature_name=feature,
                                        clazz=clazz,
                                        file_path=file_path,
                                        is_problematic=feature
                                        in problems.get(clazz, []),
                                    )
                                    html_src.img(
                                        src=(
                                            persistence.get_relative_path(file_path)
                                            if not absolute_paths
                                            else file_path
                                        ),
                                        style=(
                                            "height: 100%; max-width: 100%; "
                                            "display: block; margin: auto;"
                                        ),
                                        alt=(
                                            f"PD for class '{clazz}' and feature "
                                            f"'{feature}"
                                        ),
                                    )

            html_format.update_data(
                str(html_src),
                f"{persistence.FILE_EXPLANATION}.html",
            )

            html_explanation.add_format(html_format)

            return html_explanation

        raise NotImplementedError(
            f"Creation of the HTML fragment explanation from the {type(explanation)} "
            f"explanation is not supported"
        )


class Global3dDataExplanation(_explanations_base.Explanation):
    """Explanation with per class and feature data frames for rendering of 3D charts
    like:

    - 3D bar chart
    - heatmap

    """

    _explanation_type = "3d-data"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class AutoReportExplanation(_explanations_base.Explanation):
    """AutoReport explanation provides various document format (Word, Markdown,...)
    explanations.

    """

    _explanation_type = "auto-report"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class ModelValidationResultExplanation(_explanations_base.Explanation):
    """Model validation result explanation is (archived) tree of directories and
    documents created by an H2O MV based explainer.

    """

    _explanation_type = "model-validation-result"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class WorkDirArchiveExplanation(_explanations_base.Explanation):
    """Explainer work directory explanation provides various work dir archive
    representations like ``zip`` or ``tgz``.

    """

    _explanation_type = "work-dir-archive"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class CustomArchiveExplanation(_explanations_base.Explanation):
    """Explainer archive representation like ``zip`` or ``tgz``."""

    _explanation_type = "custom-archive"
    _is_global = True

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class ProxyExplanation(_explanations_base.Explanation):
    """Proxy explanation is provided by parent explainers."""

    _explanation_type = "proxy_explanation"

    def __init__(
        self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        _explanations_base.Explanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None
