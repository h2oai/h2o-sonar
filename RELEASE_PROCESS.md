# Release Process
H2O Sonar **release process** guidelines:

* [Release Process](#release-process)
    * [Release CHANGELOG.md template](#release-changelog-md-template)
    * [GitHub PR Template](#github-pr-template)
    * [GitHub Release template](#github-release-template)


# Release Process
```
     branch:main                             v2.3.0 tag created by release
  ---+---------------------------------------+-------
     |                                      / v2.3.0
     +- branch:dev-2.3.0 ------+-----------+
        |                     /
        +- branch:feat-42/n -+
           ^ features merged via PRs to dev branch
```

* `main` always contain **stable** H2O Sonar w/ latest release in `HEAD`


**Development and release process**:

* DEVELOPMENT branch `dev-m.m.p` is created on top of `main` for development
    * feature/bugfix/* branches...
        * ... are created on top of `dev-m.m.p`
        * ... are merged via PR to `dev-m.m.p`
        * ... maintain `CHANGELOG.md` w/ incomming changes
* RELEASE:
    * preconditions:
        * source prepared in the `dev-m.m.p` branch
        * latest `main` merged to `dev-m.m.p`
        * `CHANGELOG.md` up to date w/ correct date & branch number
    * create release **PR** for `dev-` branch to do release build
        * iterate until `dev-m.m.p` tests are green
        * make sure `CHANGELOG.md` is correct & PR description is up to date
        * **merge** the PR ... which makes `main` latest & greatest with green tests
    * GitHub release:
        * create new **release** using the GitHub UI
        * leave release **description empty** (will be generated later)
        * configure release to create **tag** `vm.m.p` in the `main` **!**
        * **publish** the release
            - it will create **tag**
            - tag triggers GitHub Actions workflow which:
                - builds **.whl** from the green tests SHA
                - publishes .whl to **S3**
                - (generates release text as downloadable artifacts)
        * release **description**:
            - edit release
            - use `Generate release notes` button to generate description
            - add **CHANGELOG** section (see below)
            - add **footnote** w/:
                - `Wheels | PR | Build | Project | 7902080`
                - ...
                - .whl S3 location
                - release PR URL
                - build URL
                - SHA
                - GitHub project URL
* STATE:
    - `main` has latest and greatest
    - .whl(s) uploaded to S3
    - GitHub release has release metadata

Branch naming conventions:

```
dev-3.0.0
```


## Release CHANGELOG.md template
Template of the `CHANGELOG.md` section for the release:

```
## [v3.?.?](https://github.com/h2oai/h2o-sonar/tree/v3.?.?) — 2026/?/?

This is a minor H2O Sonar release.

### Added

* **Evaluators**:
    * .
* **Features**:
    * .
* **Enhancements**:
    * .
* **Documentation**
    * .

### Fixed

No fixes.

### Changed

No changes.

### Deprecated

No deprecations.

### Removed

No removals.

### Security

No security fixes.
```

Use the following command to append it:

```
make releng_changelog_md_new
```


## GitHub PR Template
Template of the GitHub PR for the release:

---

H2O Sonar 3.0.0

...

This PR brings **H2O Sonar 3.0.0**.

* Changelog:
    * https://github.com/h2oai/h2o-sonar/blob/dev-2.3.0/CHANGELOG.md
* Release PR:
    * https://github.com/h2oai/h2o-sonar/pull/1201
* Python wheels:
    * https://BUCKET.s3.REGION.amazonaws.com/releases/h2o_sonar/h2o_sonar-*.whl
* Documentation:
    * https://h2oai.github.io/h2o-sonar/h2o-eval-studio/index.html
* Project:
    * https://github.com/orgs/h2oai/projects/...
* QA test report:

```
make test_generative
== ...
make test_predictive
== ...

CONFIGURATION:
$ hostname; ls deps ; ls tests/lib/given_generative*.* ; ls ~/.driverlessai 
...
```

---

GitHub **generated release notes**:

* how & where to do tags correctly:
    * tags MUST be made in `main`
        - Releases editor has `Generate release notes` button
          which will work in this case
        -https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes
    * release notes @ release editor are generated as follows:
        - **Full Changelog**:
          https://github.com/h2oai/h2o-sonar/compare/v1.0.0...v2.2.0
        - in other words, tags NOT in the main are causing problems
          to generate the release notes automatically

## GitHub Release template
Template of the GitHub Release:

```
H2O Sonar 3.0.0

...

This is a major/minor/patch H2O Sonar release.

[insert CHANGELOG.md release section]

---

[insert GitHub generated release notes section]

---

[Python wheels](https://REGION.console.aws.amazon.com/s3/...) | [Project](https://github.com/orgs/h2oai/projects/...) | [PR](https://github.com/h2oai/h2o-sonar/pull/...) | [Build](https://github.com/h2oai/h2o-sonar/actions/runs/...) | [SHA](https://github.com/h2oai/h2o-sonar/pull/.../commits/790...)
```

GitHub **generated release notes**:

* How & where to do tags correctly:
    * Tags MUST be made in `main`
        - Releases editor has `Generate release notes` button
          which will work in this case
        - https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes
    * Release notes @ release editor are generated as follows:
        - **Full Changelog**:
          https://github.com/h2oai/h2o-sonar/compare/v1.0.0...v2.2.0
        - In other words, tags NOT in the main are causing problems
          to generate the release notes automatically.


