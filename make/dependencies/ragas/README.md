# Patch of ragas 0.1.7 library 

H2O Sonar uses `ragas` library `0.1.7` that has `langchain-core`.
However, this version of `langchain-core` is vulnerable `CVE-2026-26013`.
Therefore `ragas 0.1.7` was patched to upgrade it to `langchain-core 1.2.11`.

RAGAs library:

* https://pypi.org/project/ragas/0.1.7
* https://github.com/vibrantlabsai/ragas

Changes:

* 0001-fix-ci-some-linting-issues-852.patch
   * Patch to be applied to the source code of `ragas 0.1.7`.
* ragas-v0.1.7.diff
   * Overview of changes.
   
Patched wheel:

* https://eval-studio-artifacts.s3.us-east-1.amazonaws.com/dependencies/ragas/ragas-0.1.7%2Bh2osonar.1-py3-none-any.whl


