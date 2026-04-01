# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import enum
import json
import os
import pathlib
import shutil

import pytest
import requests

from h2o_sonar import loggers
from h2o_sonar.utils import testing as sonar_testing


#
# Test Suite Library Librarian EXPERIMENTAL - PRODUCTION version can be found in:
#
#     {GITHUB}:h2oai/eval-studio/(worker)

# test suite library index file path
_LIB_INDEX_PATH = (
    pathlib.Path()
    / "data"
    / "generative"
    / "evals_library"
    / "h2o-eval-studio-suite-library.json"
)


def is_library_index_available() -> bool:
    """Check if the test suite library index file is available.

    Returns
    -------
    bool
        True if the library index file exists, False otherwise.
    """
    return _LIB_INDEX_PATH.exists()


class SamplingMethod(enum.Enum):
    HEAD = "head"
    RANDOM = "random"
    TAIL = "tail"


class PocTestSuiteLibraryLibrarian:
    """Test suite library librarian prototype - production version can be found
    in the H2O Eval Studio code base.

    """

    FILE_INDEX = "index"
    FILE_INDEX_JSON = f"{FILE_INDEX}.json"
    FILE_INDEX_MD = f"{FILE_INDEX}.md"
    FILE_INDEX_HTML = f"{FILE_INDEX}.html"

    KEY_TEST_SUITES = "test_suites"

    # IMPROVE make it data class
    class TestSuiteLibItem:
        KEY_NAME = "name"
        KEY_DESCRIPTION = "description"
        KEY_TEST_SUITE_URL = "test_suite_url"
        KEY_REFERENCE_URL = "reference_url"
        KEY_SOURCE_URL = "source_url"
        KEY_LICENSE = "license"
        KEY_EVALUATES = "evaluates"
        KEY_PURPOSES = "purposes"
        KEY_CATS = "categories"
        KEY_ORIGIN = "origin"
        KEY_TC_COUNT = "test_case_count"
        KEY_T_COUNT = "test_count"

        def __init__(self):
            self.name = ""
            self.description = ""
            self.test_suite_url = ""
            self.reference_url = ""
            self.source_url = ""
            self.license = ""
            self.evaluates = []
            self.purposes = []
            self.categories = []
            self.origin = ""
            self.test_case_count = 0
            self.test_count = 0

        @staticmethod
        def from_dict(d: dict):
            t_item = PocTestSuiteLibraryLibrarian.TestSuiteLibItem

            ts = PocTestSuiteLibraryLibrarian.TestSuiteLibItem()
            ts.name = d.get(t_item.KEY_NAME, "")
            ts.description = d.get(t_item.KEY_DESCRIPTION, "")
            ts.test_suite_url = d.get(t_item.KEY_TEST_SUITE_URL, "")
            ts.reference_url = d.get(t_item.KEY_REFERENCE_URL, "")
            ts.source_url = d.get(t_item.KEY_SOURCE_URL, "")
            ts.license = d.get(t_item.KEY_LICENSE, "")
            ts.evaluates = d.get(t_item.KEY_EVALUATES, [])
            ts.purposes = d.get(t_item.KEY_PURPOSES, [])
            ts.categories = d.get(t_item.KEY_CATS, [])
            ts.origin = d.get(t_item.KEY_ORIGIN, "")
            ts.test_case_count = d.get(t_item.KEY_TC_COUNT, 0)
            ts.test_count = d.get(t_item.KEY_T_COUNT, 0)

            return ts

    @staticmethod
    def get_cache_dir_path(cache_base_dir: str | pathlib.Path) -> pathlib.Path | None:
        """Get configured path to the librarian cache directory.

        Returns
        -------
        pathlib.Path | None :
            Path to the library cache directory.
        """
        if cache_base_dir:
            cache_base_dir = pathlib.Path(cache_base_dir)
            return cache_base_dir / "h2o_eval_studio_prompt_library"
        return None

    @staticmethod
    def get_index_path(cache_base_dir: str | pathlib.Path) -> pathlib.Path:
        """Get configured path to the (cached) library index."""
        cache_path = PocTestSuiteLibraryLibrarian.get_cache_dir_path(cache_base_dir)
        if cache_path:
            return cache_path / PocTestSuiteLibraryLibrarian.FILE_INDEX_JSON

        raise ValueError(
            "Cache base directory must be provided to create prompt library index path."
        )

    def __init__(
        self,
        cache_base_dir: str | pathlib.Path,
        name: str = "",
        description: str = "",
        test_suites: list[TestSuiteLibItem] | None = None,
        logger=None,
    ):
        """Initialize the test suite library librarian."""
        cache_dir = PocTestSuiteLibraryLibrarian.get_cache_dir_path(cache_base_dir)
        if not cache_dir:
            raise ValueError("Cache base directory must be provided.")
        self.cache_dir: pathlib.Path = cache_dir
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.name = name or "H2O Eval Studio Evaluation Test Suite Library"
        self.description = (
            description or "H2O Eval Studio evaluation test suite library."
        )
        self.test_suites = test_suites or []
        self.logger = logger or loggers.SonarPrintLogger()

    @staticmethod
    def load(
        lib_index_path: str | pathlib.Path | None,
        lib_index_url: str | None,
        cache_base_dir: str | pathlib.Path,
        logger,
        verify: str | bool = True,
    ) -> "PocTestSuiteLibraryLibrarian":
        """Load the test suite library librarian from the given library index."""
        if not lib_index_url and not lib_index_path:
            # check whether index is cached
            if not cache_base_dir:
                raise ValueError("Library index path or URL must be provided")
            lib_index_path = PocTestSuiteLibraryLibrarian.get_index_path(cache_base_dir)
            if not lib_index_path.exists():
                raise FileNotFoundError(
                    f"Library index path and URL are not provided, and the "
                    f"cached index file does not exist: {lib_index_path}"
                )

        librarian = PocTestSuiteLibraryLibrarian(
            cache_base_dir=cache_base_dir, logger=logger
        )

        if not lib_index_path and lib_index_url:
            lib_index_path = PocTestSuiteLibraryLibrarian.get_index_path(cache_base_dir)
            logger.info(
                f"Downloading the test suite library index from {lib_index_url} to "
                f"cache as {lib_index_path} ..."
            )
            try:
                with requests.get(lib_index_url, stream=True, verify=verify) as r:
                    with open(lib_index_path, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
            except Exception as e:
                raise ValueError(
                    f"Failed to download the test suit library index from"
                    f" {lib_index_url}: {e}"
                ) from e
            logger.info(f"Test suite library index cached to {lib_index_path}")

        if lib_index_path:
            logger.info(f"Creating test suite librarian from {lib_index_path} ...")
            lib_index_path = pathlib.Path(lib_index_path)
            if not lib_index_path.exists():
                raise FileNotFoundError(
                    f"Prompt library index file does not exist: {lib_index_path}"
                )

            with open(lib_index_path) as file:
                lib_dict = json.load(file)

            librarian.test_suites = []
            for ts in lib_dict[PocTestSuiteLibraryLibrarian.KEY_TEST_SUITES]:
                librarian.test_suites.append(
                    PocTestSuiteLibraryLibrarian.TestSuiteLibItem.from_dict(ts)
                )
            logger.info(
                f"Created test suite librarian with {len(librarian.test_suites)} "
                f"test suites."
            )

            return librarian

        raise ValueError(
            f"Library index path ({lib_index_path}) or URL ({lib_index_url}) "
            f"must be provided."
        )

    def list_test_suites(
        self,
        filter_by_categories: list[str] | None = None,
        filter_by_purposes: list[str] | None = None,
        filter_by_evaluates: list[str] | None = None,
        filter_by_origin: str | None = None,
        filter_by_test_case_count: int | None = None,
        filter_by_test_count: int | None = None,
        filter_by_fts: str | None = None,
    ) -> list:
        """List and optionally filter test suites in the library.

        Parameters
        ----------
        filter_by_categories : list[str] | None
            Filter by categories. Defaults to None.
        filter_by_purposes : list[str] | None
            Filter by purposes. Defaults to None.
        filter_by_evaluates : list[str] | None
            Filter by evaluates. Defaults to None.
        filter_by_origin : str | None
            Filter by origin. Defaults to None.
        filter_by_test_case_count :
            Filter by test case count - test suite must have at least the given count.
            Defaults to None.
        filter_by_test_count :
            Filter by test count - test suite must have at least the given count.
            Defaults to None.
        filter_by_fts :
            Filter by full text search of the test suite name and description.
            Defaults to None.

        Returns
        -------
        list :
            List of test suites in the library.

        """
        result = []
        for ts in self.test_suites:
            if filter_by_categories and not any(
                cat in ts.categories for cat in filter_by_categories
            ):
                continue
            if filter_by_purposes and not any(
                pur in ts.purposes for pur in filter_by_purposes
            ):
                continue
            if filter_by_evaluates and not any(
                ev in ts.evaluates for ev in filter_by_evaluates
            ):
                continue
            if filter_by_origin and ts.origin != filter_by_origin:
                continue
            if (
                filter_by_test_case_count
                and ts.test_case_count < filter_by_test_case_count
            ):
                continue
            if filter_by_test_count and ts.test_count < filter_by_test_count:
                continue
            if filter_by_fts and filter_by_fts not in f"{ts.name} {ts.description}":
                continue

            result.append(ts)

        return result

    def _get_test_suite_cache_path(self, test_suite_url: str) -> pathlib.Path:
        """Get the cache path for the test suite."""
        if not test_suite_url:
            raise ValueError("Test suite URL is not provided.")
        tokens = test_suite_url.split("/")
        if not tokens:
            raise ValueError("Invalid test suite URL.")
        return self.cache_dir / tokens[-1]

    @staticmethod
    def _download_test_suite(
        test_suite_url: str, test_suite_cache_path: pathlib.Path
    ) -> pathlib.Path:
        """Download the test suite from the given URL."""
        with requests.get(test_suite_url, stream=True) as r:
            with open(test_suite_cache_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

        return test_suite_cache_path

    def sample_test_suite(
        self,
        test_suite_url: str,
        test_case_count: int = 5,
        method: SamplingMethod = SamplingMethod.HEAD,
    ) -> dict:
        """Randomly choose given number of test cases from the test suite."""
        try:
            if not test_suite_url:
                raise ValueError("Test suite URL is not provided.")
            if not any(ts.test_suite_url == test_suite_url for ts in self.test_suites):
                raise ValueError(
                    f"Test suite {test_suite_url} not found in the prompt library."
                )
            if method != SamplingMethod.HEAD:
                raise NotImplementedError("Only 'head' sampling method is supported.")

            # check the existence of the test suite in the cache
            test_suite_cache_path = self._get_test_suite_cache_path(test_suite_url)

            # if not, download the test suite
            if not test_suite_cache_path.exists():
                PocTestSuiteLibraryLibrarian._download_test_suite(
                    test_suite_url=test_suite_url,
                    test_suite_cache_path=test_suite_cache_path,
                )

            # load the test suite from the cache
            with open(test_suite_cache_path) as file:
                test_suite_dict = json.load(file)

            # sample the test cases
            # - find the test with sufficient number of test cases
            # - if found, sample the test cases
            # - if not, return test w/ highest number of test cases (all of them)
            # - if it is RAG test suite, KEEP the corpus in the test suite
            ts_keys = sonar_testing.RagTestSuiteConfig
            sampled_test_suite = {
                ts_keys.KEY_NAME: test_suite_dict[ts_keys.KEY_NAME],
                ts_keys.KEY_DESCRIPTION: test_suite_dict[ts_keys.KEY_DESCRIPTION],
                ts_keys.KEY_TESTS: [],
            }
            highest_cardinality_test = None
            highest_cardinality = 0
            for test in test_suite_dict[ts_keys.KEY_TESTS]:
                cardinality = len(test.get(ts_keys.KEY_TEST_CASES, []))
                if cardinality >= test_case_count:
                    highest_cardinality_test = test
                    break
                if cardinality > highest_cardinality:
                    highest_cardinality_test = test
                    highest_cardinality = cardinality

            if highest_cardinality_test:
                sampled_test_suite[ts_keys.KEY_TESTS].append(highest_cardinality_test)
                sampled_test_suite[ts_keys.KEY_TESTS][0][ts_keys.KEY_TEST_CASES] = (
                    highest_cardinality_test[ts_keys.KEY_TEST_CASES][:test_case_count]
                )
                return sampled_test_suite
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Failed to sample the test suite {test_suite_url}: {e}")
            return {}


@pytest.mark.parametrize(
    "test_suites_to_sample",
    [
        5,
        # 0,  # downloads 500MB and tests all 250+ test suites (GHA traffic $)
    ],
)
@pytest.mark.skipif(
    not is_library_index_available(),
    reason="Test suite library index file not available",
)
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_librarian(tmp_path, test_suites_to_sample):
    #
    # GIVEN
    #
    test_case_count = 7
    lib_index_path = _LIB_INDEX_PATH
    work_dir_path = tmp_path / "work_dir"
    work_dir_path.mkdir(parents=True, exist_ok=True)

    #
    # WHEN
    #
    librarian = PocTestSuiteLibraryLibrarian.load(
        lib_index_path=lib_index_path,
        lib_index_url=None,
        cache_base_dir=work_dir_path,
        logger=loggers.SonarPrintLogger(),
    )

    # list test suites
    ts_list = librarian.list_test_suites()
    assert len(ts_list)
    print("\nTest suites in the library:")
    for ts_entry in ts_list:
        print(ts_entry.name)
        print(ts_entry.test_suite_url)

    # sample test suite - test them ALL
    test_suites_to_sample = test_suites_to_sample or len(ts_list)
    for i in range(test_suites_to_sample):
        ts_entry = ts_list[i]
        print(f"\nSampling test suite:\n{ts_entry.test_suite_url}")
        ts_sampled = librarian.sample_test_suite(
            test_suite_url=ts_entry.test_suite_url,
            test_case_count=test_case_count,
        )
        print(f"\nSampled test suite {i} / {ts_entry.test_suite_url}:")
        print(json.dumps(ts_sampled, indent=2))
        assert ts_sampled
        assert len(ts_sampled["tests"]) == 1
        assert len(ts_sampled["tests"][0]["test_cases"]) > 0
        ts_sampled_path = tmp_path / f"sampled_test_suite_{i}.json"
        with open(ts_sampled_path, "w") as file:
            json.dump(ts_sampled, file)

        #
        # THEN
        #
        rag_ts = sonar_testing.RagTestSuiteConfig.load_from_json(ts_sampled_path)
        print(rag_ts)
        assert rag_ts
        assert len(rag_ts.test_cases) > 0  # at least one test case


def _extract_field(field: str, d: str, is_url: bool = True) -> tuple[str, int]:
    print(f" Extracting field: '{field}'...")
    extracted_url = ""
    rest = ""

    strstr = f"\n\n{field} "
    begin = d.find(strstr)
    if begin == -1:
        strstr = f"\n{field}: "
        begin = d.find(strstr)
        if begin > -1:
            rest = d[begin + len(strstr) :]
    else:
        rest = d[begin + len(strstr) :]

    if rest:
        print(f"    Rest: '{rest}'")
        if "\n" not in rest and " " not in rest:
            extracted_url = rest
        else:
            end = rest.find("\n")
            if end == -1:
                end = rest.find(" ")
                if end > -1:
                    extracted_url = rest[:end]
            else:
                extracted_url = rest[:end]

    # validate URL
    if is_url:
        if extracted_url:
            print(f"    Extracted URL: '{extracted_url}'")
            if (
                extracted_url
                and extracted_url.startswith("http")
                and " " not in extracted_url
                and "\n" not in extracted_url
            ):
                print(f"  Valid URL: {extracted_url}")

                # return: extracted URL + end of description
                return extracted_url, begin
    elif extracted_url:
        print(f"    Extracted license: '{extracted_url}'")
        extracted_url = (
            extracted_url[:-1] if extracted_url.endswith(".") else extracted_url
        )
        return extracted_url, begin

    return "", 0


@pytest.mark.skip("Tool for extracting fields from description")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_extract_fields_from_description(tmp_path):
    #
    # GIVEN
    #

    index_path = _LIB_INDEX_PATH
    t_lib_item = PocTestSuiteLibraryLibrarian.TestSuiteLibItem
    # load index
    with open(index_path) as file:
        index_json = json.load(file)

    #
    # WHEN
    #

    new_index_json = {
        t_lib_item.KEY_NAME: index_json.get(t_lib_item.KEY_NAME, ""),
        t_lib_item.KEY_DESCRIPTION: index_json.get(t_lib_item.KEY_DESCRIPTION, ""),
        PocTestSuiteLibraryLibrarian.KEY_TEST_SUITES: [],
    }

    # extract fields from description
    for ts in index_json[PocTestSuiteLibraryLibrarian.KEY_TEST_SUITES]:
        print(f"\nTest suite: {ts[t_lib_item.KEY_NAME]}")
        d = ts[t_lib_item.KEY_DESCRIPTION]
        print(f"  >>>{d}<<<")

        if d:
            # reference URL
            (reference_url, reference_description_end) = _extract_field("Reference", d)

            # source URL
            (source_extracted_url, source_description_end) = _extract_field("Source", d)

            # license URL
            (license_extracted, license_description_end) = _extract_field(
                "License", d, is_url=False
            )

            non_0 = [
                v
                for v in [
                    reference_description_end,
                    source_description_end,
                    license_description_end,
                ]
                if v > 0
            ]
            description_end = min(non_0) if non_0 else 0
            if description_end:
                new_description = d[:description_end].strip()
            else:
                new_description = d

            print(
                f"  EXTRACTION results:\n"
                f"    {reference_url=}\n"
                f"    {source_extracted_url=}\n"
                f"    {license_extracted=}\n"
                f"    {new_description=}"
            )

            if "test_suite_resources_url" in ts:
                del ts["test_suite_resources_url"]
            if "url" in ts:
                del ts["url"]

            reference_url = reference_url or ts.get(t_lib_item.KEY_REFERENCE_URL, "")
            ts[t_lib_item.KEY_REFERENCE_URL] = reference_url

            source_extracted_url = source_extracted_url or ts.get(
                t_lib_item.KEY_SOURCE_URL, ""
            )
            ts[t_lib_item.KEY_SOURCE_URL] = source_extracted_url

            license_extracted = license_extracted or ts.get(t_lib_item.KEY_LICENSE, "")
            ts[t_lib_item.KEY_LICENSE] = license_extracted

            if new_description:
                ts[t_lib_item.KEY_DESCRIPTION] = new_description

            new_index_json[PocTestSuiteLibraryLibrarian.KEY_TEST_SUITES].append(ts)

        else:
            new_index_json[PocTestSuiteLibraryLibrarian.KEY_TEST_SUITES].append(ts)

    new_index_path = tmp_path / "new_index.json"
    with open(new_index_path, "w") as f:
        json.dump(new_index_json, f, indent=2)

    #
    # THEN
    #
    print(f"\nNew index saved to: file://{new_index_path}")


@pytest.mark.skip("Tool for test suite library licenses maintenance")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_maintain_licenses(tmp_path):
    #
    # GIVEN
    #

    index_path = _LIB_INDEX_PATH
    t_lib_item = PocTestSuiteLibraryLibrarian.TestSuiteLibItem
    # load index
    with open(index_path) as file:
        index_json = json.load(file)

    #
    # WHEN
    #

    missing_licenses = []
    for ts in index_json[PocTestSuiteLibraryLibrarian.KEY_TEST_SUITES]:
        print(f"Test suite: {ts[t_lib_item.KEY_NAME]}")

        if t_lib_item.KEY_REFERENCE_URL not in ts:
            print(json.dumps(ts, indent=2))
            raise RuntimeError(
                f"  Missing {t_lib_item.KEY_REFERENCE_URL}: {ts[t_lib_item.KEY_NAME]}"
            )
        if t_lib_item.KEY_SOURCE_URL not in ts:
            raise RuntimeError(
                f"  Missing {t_lib_item.KEY_SOURCE_URL}: {ts[t_lib_item.KEY_NAME]}"
            )

        # force licenses where it's clear
        if ts[t_lib_item.KEY_REFERENCE_URL] == "https://github.com/h2oai/h2o-evals":
            ts[t_lib_item.KEY_LICENSE] = "Mozilla Public License 2.0"
        elif "moonshot-data" in ts[t_lib_item.KEY_SOURCE_URL]:
            ts[t_lib_item.KEY_LICENSE] = "Apache License 2.0"
        elif not ts[t_lib_item.KEY_LICENSE]:
            if ts[t_lib_item.KEY_ORIGIN] == "h2oai":
                ts[t_lib_item.KEY_LICENSE] = "Mozilla Public License 2.0"
            else:
                missing_licenses.append(ts[t_lib_item.KEY_NAME])

    print("\n# Test suites with missing licenses: " + 10 * "#")
    for ts_name in missing_licenses:
        print(f"  {ts_name}")

    # save updated index file
    new_index_path = tmp_path / "new_index.json"
    with open(new_index_path, "w") as f:
        json.dump(index_json, f, indent=2)

    #
    # THEN
    #
    print(f"\nNew index saved to: file://{new_index_path}")


@pytest.mark.skip("Tool for test suite library URL links validation")
@pytest.mark.generative
@pytest.mark.h2o_sonar
def test_online_prompt_library_sanity_check(tmp_path):
    """Test that the whole ONLINE version of the prompt library is accessible and...

    - all links to test suites are valid
    - test suites can be downloaded
    - test suites links to corpus documents are valid

    """
    #
    # GIVEN
    #

    p_lib_idx_url = (
        "https://eval-studio-artifacts.s3.us-east-1.amazonaws.com"
        "/h2o-eval-studio-suite-library/index-CUAD.json"
    )
    urls_skip_list = [
        "https://www.wikipedia.org/",
    ]
    cache_dir = tmp_path

    #
    # WHEN
    #

    # download the index
    print(f"INDEX: Downloading '{p_lib_idx_url}' to '{cache_dir}' ...")
    index_json = {}
    try:
        response = requests.get(p_lib_idx_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        index_path = os.path.join(cache_dir, "index.json")
        with open(index_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded '{p_lib_idx_url}' to '{index_path}'")

        # load the index
        with open(index_path) as file:
            index_json = json.load(file)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading '{p_lib_idx_url}': {e}")
        raise e
    except OSError as e:
        print(f"Error creating directory or saving file: {e}")
        raise e
    assert index_json

    # download all test suites
    print("TEST SUITES: Downloading all test suites ...")
    test_suite_infos = index_json[PocTestSuiteLibraryLibrarian.KEY_TEST_SUITES]
    print(f"  Found {len(test_suite_infos)} test suites in the index")
    assert test_suite_infos
    for e_ts, test_suite_info in enumerate(test_suite_infos):
        test_suite_url = test_suite_info.get(
            PocTestSuiteLibraryLibrarian.TestSuiteLibItem.KEY_TEST_SUITE_URL
        )
        assert test_suite_url, "'test_suite_url' key not found in the input dictionary"

        test_suite_json = {}
        try:
            print(f"{e_ts}. Downloading test suite from '{test_suite_url}' ...")
            response = requests.get(test_suite_url)
            response.raise_for_status()
            test_suite_json = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading content from '{test_suite_url}': {e}")
            raise e
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from the downloaded content: {e}")
            raise e
        assert test_suite_json

        print(
            f"{e_ts}. CORPUS: Checking corpus entries URLs for test suite"
            f" {test_suite_url} ..."
        )
        tc_tests = test_suite_json.get(sonar_testing.RagTestSuiteConfig.KEY_TESTS, [])
        assert tc_tests, "Test suite does not contain any tests"
        for e_t, t in enumerate(tc_tests):
            t_docs = t.get(sonar_testing.RagTestConfig.KEY_DOCUMENTS, [])
            if not t_docs:
                print(f"  No documents in a test case of {test_suite_url}")
                continue

            for e_d, doc_url in enumerate(t_docs):
                assert doc_url, f"Invalid document URL: '{doc_url}'"
                if doc_url in urls_skip_list:
                    print(f"  Skipping document URL '{doc_url}' ...")
                    continue
                print(
                    f"  {e_ts}.{e_t}.{e_d}. Checking document URL validity for "
                    f"'{doc_url}' ..."
                )
                try:
                    response = requests.head(doc_url, allow_redirects=True, timeout=5)
                    response.raise_for_status()
                except requests.exceptions.RequestException:
                    raise RuntimeError(
                        f"Document URL '{doc_url}' for test suite {test_suite_url} is "
                        f"not valid or not accessible (skipped URLs: {urls_skip_list})"
                    )

    #
    # THEN
    #

    print("All test suites downloaded and corpus URLs checked successfully.")


# override H2O-3 fixture as it is not desired in this test module
@pytest.fixture(autouse=True)
def h2o3_init_fixture():
    return
