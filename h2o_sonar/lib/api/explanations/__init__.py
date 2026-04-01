# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
from h2o_sonar.lib.api.explanations._explanations import AutoReportExplanation
from h2o_sonar.lib.api.explanations._explanations import CustomArchiveExplanation
from h2o_sonar.lib.api.explanations._explanations import DiaExplanation
from h2o_sonar.lib.api.explanations._explanations import Global3dDataExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalDataFrameExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalDtExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalFeatImpExplanation
from h2o_sonar.lib.api.explanations._explanations import (
    GlobalGroupedBarChartExplanation,
)
from h2o_sonar.lib.api.explanations._explanations import GlobalHtmlFragmentExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalLinePlotExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalNlpLocoExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalRuleExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalScatterPlotExplanation
from h2o_sonar.lib.api.explanations._explanations import GlobalSummaryFeatImpExplanation
from h2o_sonar.lib.api.explanations._explanations import (
    IndividualConditionalExplanation,
)
from h2o_sonar.lib.api.explanations._explanations import LocalDataFrameExplanation
from h2o_sonar.lib.api.explanations._explanations import LocalDtExplanation
from h2o_sonar.lib.api.explanations._explanations import LocalFeatImpExplanation
from h2o_sonar.lib.api.explanations._explanations import LocalHtmlSnippetExplanation
from h2o_sonar.lib.api.explanations._explanations import LocalNlpLocoExplanation
from h2o_sonar.lib.api.explanations._explanations import LocalRuleExplanation
from h2o_sonar.lib.api.explanations._explanations import LocalSummaryFeatImpExplanation
from h2o_sonar.lib.api.explanations._explanations import LocalTextSnippetExplanation
from h2o_sonar.lib.api.explanations._explanations import LocoExplanation
from h2o_sonar.lib.api.explanations._explanations import (
    ModelValidationResultExplanation,
)
from h2o_sonar.lib.api.explanations._explanations import NlpTokenizerExplanation
from h2o_sonar.lib.api.explanations._explanations import OnDemandExplanation
from h2o_sonar.lib.api.explanations._explanations import PartialDependenceExplanation
from h2o_sonar.lib.api.explanations._explanations import ProxyExplanation
from h2o_sonar.lib.api.explanations._explanations import ReportExplanation
from h2o_sonar.lib.api.explanations._explanations import SaExplanation
from h2o_sonar.lib.api.explanations._explanations import TextExplanation
from h2o_sonar.lib.api.explanations._explanations import TimeSeriesAppExplanation
from h2o_sonar.lib.api.explanations._explanations import WorkDirArchiveExplanation
from h2o_sonar.lib.api.explanations._explanations_base import Explanation
from h2o_sonar.lib.api.explanations._explanations_base import ExplanationDescriptor
from h2o_sonar.lib.api.explanations._explanations_base import SentenceComparisonMethod
from h2o_sonar.lib.api.explanations._explanations_cmp import EvalResultDiff
from h2o_sonar.lib.api.explanations._explanations_cmp import EvalResultsDiff
from h2o_sonar.lib.api.explanations._explanations_cmp import (
    EvalResultsExplanationsComparator,
)
from h2o_sonar.lib.api.explanations._explanations_genai import LlmEvalResultsExplanation
from h2o_sonar.lib.api.explanations._explanations_leaderboards import (
    AbcHeatmapExplanation,
)
from h2o_sonar.lib.api.explanations._explanations_leaderboards import DurationStatsKey
from h2o_sonar.lib.api.explanations._explanations_leaderboards import (
    LlmBoolLeaderboardExplanation,
)
from h2o_sonar.lib.api.explanations._explanations_leaderboards import (
    LlmClassifierLeaderboardExplanation,
)
from h2o_sonar.lib.api.explanations._explanations_leaderboards import (
    LlmHeatmapLeaderboardExplanation,
)
from h2o_sonar.lib.api.explanations._explanations_leaderboards import (
    LlmLeaderboardExplanation,
)
from h2o_sonar.lib.api.explanations._explanations_leaderboards import (
    LlmProcedureEvalLeaderboardExplanation,
)
from h2o_sonar.lib.api.explanations._explanations_perturbations import (
    diagnose_perturbation_flips,
)
from h2o_sonar.lib.api.explanations._explanations_perturbations import (
    FlippedPerturbedTestCase,
)
