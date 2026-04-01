# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
"""Get versions, licenses, GitHub repositories for the project dependencies."""

import json
import os
import subprocess

import requests


# packages which are used for building and testing the project and/or our packages
BLACKLIST = {
    "pip",
    "flake8",
    "flake8-import-order",
    "pytest",
    "isort",
    "black",
    "mypy",
    "setuptools",
    "h2ogpte",
    "h2o-authn",
}

METHOD_PKGS_FREEZE = "pip freeze"
METHOD_PKGS_LIST = "pip list"

# metadata of problematic packages (pypi data is not available or broken):
KNOWN_META = {
    # https://pypi.org/pypi/ragas/json
    "ragas": {
        "license": "Apache 2.0",
        "github": "https://github.com/explodinggradients/ragas/",
    },
    # https://pypi.org/pypi/bs4/json
    "bs4": {
        "license": "MIT License",
        "github": "https://code.launchpad.net/beautifulsoup",
    },
    # https://pypi.org/pypi/beautifulsoup4/json
    "beautifulsoup4": {
        "license": "MIT License",
        "github": "https://code.launchpad.net/beautifulsoup",
    },
    "h2ogpte": {
        "license": "H2O.ai License",
        "github": "https://h2oai.github.io/h2ogpte/index.html",
    },
}


def get_installed_packages(method=METHOD_PKGS_FREEZE) -> tuple[list[tuple], list]:
    # result is a list of tuples:
    # (package_name, package_version, package_license, GitHub repository URL)
    result = []
    failed_packages = []

    # executable = sys.executable
    executable = "python3"

    if method == METHOD_PKGS_FREEZE:
        cli_cmd = [executable, "-m", "pip", "freeze"]
    elif method == METHOD_PKGS_LIST:
        cli_cmd = [executable, "-m", "pip", "list", "--format=json"]
    else:
        raise ValueError(f"Unknown method: {method}")

    child_env = os.environ.copy()
    # child_env["PYTHONPATH"] = "."

    print(f"\nRunning command: {cli_cmd}")
    p = subprocess.Popen(
        cli_cmd, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    print(f"\nOutput: {out.decode()}")

    assert p.wait() == 0

    # read output line by line and parse it
    for line in out.decode().splitlines():
        print(f"\nLine: {line}")
        if "@" in line:
            # non-pip package
            print(f"Skipping non-pip package: {line}")
            continue
        else:
            try:
                package_name, package_version = line.split("==")
            except ValueError:
                failed_packages.append(line)
                raise ValueError(f"Cannot parse line: {line}")
            print(f"Package: {package_name}, Version: {package_version}")
            result.append((package_name, package_version, None, None))

    return result, failed_packages


def get_package_meta_from_pypi(package_tuples: list[tuple]) -> tuple[list[tuple], list]:
    """Get package metadata from PyPI using name and version."""
    failed_github_urls = []
    package_tuples_complete = []

    for package_name, package_version, package_license, github_url in package_tuples:
        print(f"\nPackage: {package_name}, Version: {package_version}")

        if package_name in BLACKLIST:
            print(f"Skipping blacklisted package: {package_name}")
            continue

        if package_name in KNOWN_META:
            package_license = KNOWN_META[package_name]["license"]
            github_url = KNOWN_META[package_name]["github"]
            package_tuples_complete.append(
                (package_name, package_version, package_license, github_url)
            )
            continue

        pypi_url = f"https://pypi.org/pypi/{package_name}/{package_version}/json"
        print(f"PyPI URL: {pypi_url}")
        with requests.get(pypi_url) as response:
            data = response.text
            package_info = json.loads(data)
            # print(f"Package info: {package_info}")

            # license
            package_license = None
            if package_name.startswith("nvidia-"):
                package_license = (
                    "NVIDIA EULA https://docs.nvidia.com/cuda/eula/index.html"
                )
            if not package_license:
                pkg_classifiers = package_info.get("info", {}).get("classifiers", [])
                for c in pkg_classifiers:
                    if c.startswith("License"):
                        package_license = c.split(" :: ")[-1]
                        break
            if not package_license:
                package_license = package_info.get("info", {}).get("license")
                if package_license and package_license.startswith("MIT "):
                    package_license = "MIT License"

            print(f"License: {package_license}")

            # GitHub URL
            github_url = None
            project_urls = package_info.get("info", {}).get("project_urls", {})
            if project_urls:
                for k in [
                    "Repository",
                    "repository",
                    "Source",
                    "source",
                    "GitHub: repo",
                    "Code",
                    "code",
                ]:
                    if k in project_urls:
                        github_url = project_urls[k]
                        break
                if not github_url:
                    for u in project_urls.values():
                        if "github" in u:
                            github_url = u
                            break
            if not github_url:
                github_url = package_info.get("info", {}).get("home_page")
            if not github_url:
                failed_github_urls.append(package_name)
            print(f"GitHub URL: {github_url}")
            package_tuples_complete.append(
                (package_name, package_version, package_license, github_url)
            )

    return package_tuples_complete, failed_github_urls


def package_tuples_to_csv(
    package_tuples: list[tuple], csv_path="dependencies.csv"
) -> str:
    with open(csv_path, "w") as f:
        f.write("Package,Version,License,GitHub\n")
        for (
            package_name,
            package_version,
            package_license,
            github_url,
        ) in package_tuples:
            f.write(
                f"{package_name},{package_version},{package_license},{github_url}\n"
            )
    return csv_path


if __name__ == "__main__":
    pkg_tuples_2, fail_parse = get_installed_packages()
    pkg_tuples_4, fail_meta = get_package_meta_from_pypi(pkg_tuples_2)
    csv_path = package_tuples_to_csv(pkg_tuples_4)

    failed_packages = fail_parse + fail_meta
    print(f"\nFailed packages: {failed_packages}")
    print(f"\nSaved dependencies to: {csv_path}")
