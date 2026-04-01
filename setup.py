# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import glob
import multiprocessing as mp
import os
import sys

import setuptools

# Cython must be imported after setuptools to prevent error during the build
from Cython import Build  # noqa: I100,I202
from setuptools.command import build_py


PROJECT_NAME = "h2o_sonar"
DAI_DISTRO = False

do_regular_wheel = True


#
# Custom build steps
#


class CustomBuildPyCommand(build_py.build_py):
    def run(self):
        # package data files but not .py files
        build_py.build_py.build_package_data(self)

        # discover all package directories from extension modules
        package_dirs = set()
        if hasattr(self.distribution, "ext_modules") and self.distribution.ext_modules:
            for ext in self.distribution.ext_modules:
                # extract package path from extension name
                # (e.g., "h2o_sonar.lib.api" -> "h2o_sonar/lib/api")
                parts = ext.name.split(".")
                for i in range(1, len(parts) + 1):
                    package_dirs.add(".".join(parts[:i]))

        # also include explicitly specified packages
        package_dirs.update(self.packages)

        # create empty __init__.py in all package dirs
        for pdir in package_dirs:
            # convert package name
            # (e.g., "h2o_sonar.lib") to path (e.g., "h2o_sonar/lib")
            pdir_path = pdir.replace(".", os.sep)
            pkg_dir = os.path.join(self.build_lib, pdir_path)
            os.makedirs(pkg_dir, exist_ok=True)
            open(os.path.join(pkg_dir, "__init__.py"), "a").close()


class CustomBuildPyCommandSource(build_py.build_py):
    """build command for source (non-cythonized) wheels - includes all .py files."""

    pass  # use default build_py behavior


#
# Main setup
#


# Cache for parsed pyproject.toml to avoid repeated file I/O and parsing
_PYPROJECT_CACHE: dict | None = None


def parse_pyproject_toml() -> dict:
    """
    Parse pyproject.toml and return the data structure.

    Result is cached to avoid repeated file I/O and parsing operations.
    """
    global _PYPROJECT_CACHE

    if _PYPROJECT_CACHE is not None:
        return _PYPROJECT_CACHE

    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # fallback for Python 3.10
        except ImportError:
            import toml as tomllib  # last resort fallback

    with open("pyproject.toml", "rb" if hasattr(tomllib, "load") else "r") as f:
        _PYPROJECT_CACHE = tomllib.load(f)

    return _PYPROJECT_CACHE


def get_test_dependencies() -> list[str]:
    """Get test dependencies from pyproject.toml [dependency-groups]."""
    pyproject = parse_pyproject_toml()
    return pyproject.get("dependency-groups", {}).get("test", [])


def get_dependencies() -> list[str]:
    """Get core dependencies from pyproject.toml [project.dependencies]."""
    pyproject = parse_pyproject_toml()
    return pyproject.get("project", {}).get("dependencies", [])


def get_explainers_dependencies() -> list[str]:
    """Get explainers optional dependencies from pyproject.toml."""
    pyproject = parse_pyproject_toml()
    return (
        pyproject.get("project", {})
        .get("optional-dependencies", {})
        .get("explainers", [])
    )


def get_evaluators_dependencies() -> list[str]:
    """Get evaluators optional dependencies from pyproject.toml."""
    pyproject = parse_pyproject_toml()
    return (
        pyproject.get("project", {})
        .get("optional-dependencies", {})
        .get("evaluators", [])
    )


def get_genaiclient_dependencies() -> list[str]:
    """Get genaiclient optional dependencies from pyproject.toml."""
    pyproject = parse_pyproject_toml()
    return (
        pyproject.get("project", {})
        .get("optional-dependencies", {})
        .get("genaiclient", [])
    )


with open("README.md") as readme_markdown:
    long_description = readme_markdown.read()


# Read version
about_info = {}
with open("h2o_sonar/BUILD_INFO.txt") as f:
    exec(f.read(), about_info)

# files to be Cythonized
EXTENSIONS = glob.glob("h2o_sonar/**/*.py", recursive=True)

# number of parallel compilations
NB_COMPILE_JOBS = mp.cpu_count()


def setup_given_extensions(extensions: list):
    if extensions:
        ext_modules = Build.cythonize(
            extensions,
            compiler_directives={
                "always_allow_keywords": True,
                "language_level": str(sys.version_info[0]),
            },
        )
        src_packages = []
        cmdclass = {"build_py": CustomBuildPyCommand}
    else:
        ext_modules = []
        src_packages = setuptools.find_packages()
        cmdclass = {"build_py": CustomBuildPyCommandSource}

    if do_regular_wheel:
        extras_require = {
            "evaluators": get_evaluators_dependencies(),
            "explainers": get_explainers_dependencies(),
            "genaiclient": get_genaiclient_dependencies(),
            "testing": get_test_dependencies(),
        }
    else:
        extras_require = {"testing": get_test_dependencies()}

    setuptools.setup(
        author="H2O.ai",
        author_email="support@h2o.ai",
        cmdclass=cmdclass,
        description=(
            "H2O Sonar is a Python package that enables a holistic, low-risk, "
            "human-interpretable, fair, and trustworthy approach to machine learning "
            "by implementing various facets of Responsible predictive/generative AI."
        ),
        long_description_content_type="text/markdown",
        long_description=long_description,
        # convert .py files to .c and compile to .so
        ext_modules=ext_modules,
        extras_require=extras_require,
        install_requires=[] if DAI_DISTRO else get_dependencies(),
        name=PROJECT_NAME,
        packages=src_packages,
        package_data={"h2o_sonar": ["BUILD_INFO.txt"]},
        url="https://github.com/h2oai/h2o-sonar",
        version=about_info["version"],
        zip_safe=True,
        python_requires=">=3.8.1",
        entry_points={
            "console_scripts": ["h2o-sonar=h2o_sonar.h2o_sonar_cli:main"],
        },
    )


def setup_extensions_in_sequential(cythonize_wheel: bool):
    setup_given_extensions(EXTENSIONS if cythonize_wheel else [])


def setup_extensions_in_parallel(cythonize_wheel: bool):
    if cythonize_wheel:
        Build.cythonize(
            EXTENSIONS,
            nthreads=NB_COMPILE_JOBS,
            compiler_directives={"always_allow_keywords": True},
        )
        # explicitly tell `multiprocessing` lib to `fork` when creating a pool since
        # the macOS default is "spawn", which executes code multiple times
        pool = mp.get_context("fork").Pool(processes=NB_COMPILE_JOBS)
        pool.map(setup_given_extensions, EXTENSIONS)
        pool.close()
        pool.join()
    else:
        pool = mp.get_context("fork").Pool(processes=NB_COMPILE_JOBS)
        pool.map(setup_given_extensions, [])
        pool.close()
        pool.join()


if __name__ == "__main__":
    dai_distro = "--dai"
    regular_distro = "--regular"
    if dai_distro in sys.argv:
        PROJECT_NAME += "_dai"
        DAI_DISTRO = True
        sys.argv.remove(dai_distro)
        do_regular_wheel = False
    if regular_distro in sys.argv:
        sys.argv.remove(regular_distro)

    do_cythonize_wheel = False
    if "--cythonize" in sys.argv:
        sys.argv.remove("--cythonize")
        do_cythonize_wheel = True

    if "build_ext" in sys.argv:
        setup_extensions_in_parallel(do_cythonize_wheel)
    else:
        setup_extensions_in_sequential(do_cythonize_wheel)
