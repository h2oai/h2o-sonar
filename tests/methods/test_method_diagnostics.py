# Copyright (C) 2018-2026 H2O.ai, Inc. All rights reserved
from unittest import TestCase

from h2o_sonar import loggers as logging
from h2o_sonar.methods.utils import _method_diagnostics


class TestMethodDiagnostics(TestCase):
    """Test method diagnostics."""

    def setUp(self):
        logging.setLevel(logging.DEBUG)

    def test_inc_and_go(self):
        md = _method_diagnostics.MethodDiagnostics()

        self.assertEqual(0, md.total_scorer_calls)
        self.assertIsNotNone(md.scorer_calls_history)
        self.assertListEqual([0], md.scorer_calls_history)

        md.add_scorer_calls()
        self.assertEqual(1, md.total_scorer_calls)
        self.assertListEqual([1], md.scorer_calls_history)

        md.add_scorer_calls(2)
        self.assertEqual(3, md.total_scorer_calls)
        self.assertListEqual([3], md.scorer_calls_history)

        md.add_scorer_calls_slot()
        self.assertEqual(3, md.total_scorer_calls)
        self.assertListEqual([3, 0], md.scorer_calls_history)

        md.add_scorer_calls(5)
        self.assertEqual(8, md.total_scorer_calls)
        self.assertListEqual([3, 5], md.scorer_calls_history)

    def test_new_entry_and_go(self):
        md = _method_diagnostics.MethodDiagnostics()

        md.add_scorer_calls_slot()
        self.assertListEqual([0], md.scorer_calls_history)

        md.add_scorer_calls()
        self.assertEqual(1, md.total_scorer_calls)
        self.assertListEqual([1], md.scorer_calls_history)

    def test_overflow(self):
        md = _method_diagnostics.MethodDiagnostics()

        for i in range(_method_diagnostics.MethodDiagnostics.SCORER_HISTORY_LIMIT):
            md.add_scorer_calls_slot()
            md.add_scorer_calls(i)

        logging.debug(f"Diagnostics: {md.scorer_calls_history}")
        self.assertEqual(
            _method_diagnostics.MethodDiagnostics.SCORER_HISTORY_LIMIT - 1,
            len(md.scorer_calls_history),
        )

        md.add_scorer_calls_slot()
        md.add_scorer_calls(50)
        md.add_scorer_calls_slot()
        md.add_scorer_calls(51)
        logging.debug(f"Total: {md.total_scorer_calls}")
        logging.debug(f"Diagnostics: {md.scorer_calls_history}")
        self.assertEqual(
            _method_diagnostics.MethodDiagnostics.SCORER_HISTORY_LIMIT,
            len(md.scorer_calls_history),
        )
        self.assertEqual(2, md.scorer_calls_history[0])
