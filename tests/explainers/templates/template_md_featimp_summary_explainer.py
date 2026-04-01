# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import random

import pandas
from matplotlib import pyplot

from h2o_sonar.lib.api import explainers
from h2o_sonar.lib.api import explanations
from h2o_sonar.lib.api import formats
from h2o_sonar.lib.api import models
from h2o_sonar.lib.api import plots
from h2o_sonar.lib.api import results


class TemplateMarkdownFeatImpSummaryExplainer(explainers.Explainer):
    """Markdown report with summary feature importance chart explainer template.

    Use this template to create explainer which creates global Markdown report
    explanations.

    """

    _display_name = "Template Markdown Shapley Summary Plot"
    _description = (
        "Markdown report with summary feature importance chart explainer "
        "template which can be used to create explainer which creates global "
        "report explanations."
    )
    _regression = True
    _binary = True
    _global_explanation = True
    _explanation_types = [explanations.ReportExplanation]
    _keywords = [explainers.Explainer.KEYWORD_TEMPLATE]

    def setup(self, model: models.ExplainableModel, persistence, **kwargs):
        explainers.Explainer.setup(self, model=model, persistence=persistence, **kwargs)

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Create global and local (pre-computed/cached) explanations.

        Template explainer returns MOCK explainer data - replace mock data
        preparation with actual computation to create real explanations.

        """
        # explanations list
        model_explanations = list()

        # global explainer
        model_explanations.append(self.explain_global_markdown())

        return model_explanations

    def explain_global_markdown(self):
        global_explanation = explanations.ReportExplanation(
            explainer=self,
            display_name="Template Feature Importance Summary Markdown report",
            display_category=explanations.Explanation.DISPLAY_CAT_EXAMPLE,
        )

        # CALCULATION: Markdown report with image(s) in work directory
        report_path, images_path = self._create_report()

        # NORMALIZATION: Markdown report to Grammar of MLI format in Driverless AI UI
        global_explanation.add_format(
            formats.MarkdownFormat(
                explanation=global_explanation,
                format_file=report_path,
                extra_format_files=images_path,
            )
        )

        return global_explanation

    def get_result(self) -> results.TemplateResult:
        return results.TemplateResult(
            persistence=self.persistence,
            explainer_id=self.explainer_id(),
            explainer_name=TemplateMarkdownFeatImpSummaryExplainer._display_name,
        )

    def _create_report(self) -> tuple[str, list[str]]:
        # save image
        img_file_name = "image.png"
        work_img_path = self.persistence.get_explainer_working_file(img_file_name)
        TemplateMarkdownFeatImpSummaryExplainer._create_report_image(work_img_path)
        # save report
        report_path = self.persistence.get_explainer_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(MARKDOWN_TEMPLATE.format(img_file_name))

        return report_path, [work_img_path]

    @staticmethod
    def _create_report_image(img_path: str):
        def generate_chart_data(max_features: int = 10):
            data: dict = {}
            for i in range(max_features):
                data[f"feature_{i}"] = [random.uniform(0, 1) for _ in range(100)]
            return data

        contributions = pandas.DataFrame(generate_chart_data())
        frame = pandas.DataFrame(generate_chart_data())

        plot = plots.ScatterFeatImpPlot.plot(contributions=contributions, frame=frame)

        plot.savefig(fname=img_path)

        pyplot.savefig(img_path, dpi=300)


#
# Markdown report
#

MARKDOWN_TEMPLATE: str = """# Example Feature Importance Summary Report
This is an example of **Markdown report** which can be created by the explainer.

![image](./{})

"""
