## Fix All Test Failures

1. Run the full test suite: `python -m pytest tests/ --tb=short 2>&1 | tee /tmp/test-output.txt`
2. Parse ALL failures from output (do not stop at first)
3. For each failure, grep the codebase for similar patterns
4. Fix ALL occurrences across ALL files in one pass
5. Re-run the full test suite to verify zero failures
6. Do NOT declare done until all tests pass

Then use: /fix-all-tests
