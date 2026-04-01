# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import random

from h2o_sonar.lib.api import datasets as d6s
from h2o_sonar.lib.api import evaluators
from h2o_sonar.lib.api import explanations as e10s
from h2o_sonar.lib.api import formats as f5s


class BringYourOwnEvaluatorDemo(evaluators.Evaluator):
    _display_name = "BYOE demo"
    _description = "Bring Your Own Evaluator (BYOE) evaluator!"

    # COMPATIBILITY: RAG-hosted LLMs only
    _rag = True

    # GLOBAL: average factual consistency metric of all dataset rows
    _global_explanation = True
    _explanation_types = [e10s.LlmHeatmapLeaderboardExplanation]

    def __init__(self):
        evaluators.Evaluator.__init__(self)

    def evaluate(self, llm_testset, explanations_types: list = None, **kwargs) -> list:
        llm_dataset_as_dt = llm_testset.as_dt()

        #
        # TODO vvvvvvvvvvvvvvvvvvvvv YOUR EVALUATION CODE GOES HERE vvvvvvvvvvvvvvvvvvv
        #
        my_metric_id = "hallucination"
        scores_dict = {my_metric_id: []}
        for i in range(llm_dataset_as_dt.shape[0]):
            scores_dict[my_metric_id].append(random.uniform(0, 1))
        #
        # TODO ^^^^^^^^^^^^^^^^^^^^ YOUR EVALUATION CODE GOES HERE ^^^^^^^^^^^^^^^^^^^^
        #

        # EXPLANATIONS
        explanations = []

        # EXPLANATION: heatmap
        heatmap_explanation = e10s.LlmHeatmapLeaderboardExplanation(
            evaluator=self,
            display_name="Hallucination metrics heatmap",
            display_category=e10s.GlobalSummaryFeatImpExplanation.DISPLAY_CAT_LLM,
            key_2_rag_model={m.key: m for m in self.models},
            metric_id_2_name={my_metric_id: "Hallucination"},
            metric_id_2_threshold={m: 0.75 for m in [my_metric_id]},
            logger=self.logger,
        )
        # populate the heatmap
        key_2_llm_model = {m.key: m for m in self.models}
        for i in range(llm_dataset_as_dt.shape[0]):
            r_dict = llm_dataset_as_dt[i, :].to_dict()
            r_model = key_2_llm_model[r_dict[d6s.LlmDataset.COL_MODEL_KEY][0]]
            r_document = str(r_model.documents[0])
            r_prompt = r_dict[d6s.LlmDataset.COL_INPUT][0]

            for m in scores_dict:
                heatmap_explanation.add_col_value(
                    llm_model_name=r_model.llm_model_name or "",
                    docs=r_document,
                    prompt=r_prompt,
                    metrics_id=m,
                    value=scores_dict[m][i],
                    result_row=r_dict,
                )

        # REPRESENTATION: heat map in HTML format
        explanations.append(
            BringYourOwnEvaluatorDemo._html_explanation(
                heatmap_explanation=heatmap_explanation,
                explainer=self,
                sort_by_metric_id=my_metric_id,
                display_name="Hallucination metrics as HTML",
                display_category=e10s.Explanation.DISPLAY_CAT_COMPLIANCE,
            )
        )

        return explanations

    @staticmethod
    def _html_explanation(
        heatmap_explanation: e10s.LlmHeatmapLeaderboardExplanation,
        explainer,
        sort_by_metric_id: str,
        display_name: str = None,
        display_category: str = None,
    ):
        html_explanation = e10s.GlobalHtmlFragmentExplanation(
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )
        html_format = f5s.HtmlFormat(
            explanation=html_explanation,
            format_data=f5s.HtmlFormat.MINIMAL_HTML,
            persistence=explainer.persistence.store,
        )

        html_format.update_data(
            str(heatmap_explanation.as_html(sort_by_metric_id=sort_by_metric_id)),
            f"{explainer.persistence.FILE_EXPLANATION}.html",
        )

        html_explanation.add_format(html_format)

        return html_explanation
