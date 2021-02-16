# Contributing

## License

Intel Project for LLVM\* technology is licensed under the terms of the Apache
License v2.0 with LLVM Exceptions license ([LICENSE.txt](llvm/LICENSE.TXT)) to
ensure our ability to contribute this project to the LLVM project under the
same license.

By contributing to this project, you agree to the Apache License v2.0 with LLVM
Exceptions and copyright terms there in and release your contribution under
these terms.

## Contribution process

### Development

**NB**: For any changes not related to DPC++, but rather to LLVM in general, it
is strongly encouraged that you submit the patch to https://llvm.org/ directly.
See [LLVM contribution guidelines](https://llvm.org/docs/Contributing.html)
for more information.

**NB**: A change in compiler and runtime should be accompanied with
corresponding test changes.
See [Test DPC++ toolchain](sycl/doc/GetStartedGuide.md#test-dpc-toolchain)
section of Get Started guide for more information.

**Note (October, 2020)**: DPC++ runtime and compiler ABI is currently in frozen
state. This means that no ABI-breaking changes will be accepted by default.
Project maintainers may still approve breaking changes in some cases. Please,
see [ABI Policy Guide](sycl/doc/ABIPolicyGuide.md) for more information.

- Create a personal fork of the project on GitHub
  - For the DPC++ Compiler project, use **sycl** branch as baseline for your
    changes. See [Get Started Guide](sycl/doc/GetStartedGuide.md).
- Prepare your patch
  - follow [LLVM coding standards](https://llvm.org/docs/CodingStandards.html)
  - [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and
    [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) tools can be
    integrated into your workflow to ensure formatting and stylistic
    compliance of your changes.
  - use

    ```bash
    ./clang/tools/clang-format/git-clang-format `git merge-base origin/sycl HEAD`
    ```

    to check the format of your current changes against the `origin/sycl`
    branch.
    - `-f` to also correct unstaged changes
    - `--diff` to only print the diff without applying
- Build the project following
[Get Started Guide instructions](sycl/doc/GetStartedGuide.md#build-dpc-toolchain).
- Run regression tests -
[instructions](sycl/doc/GetStartedGuide.md#test-dpc-toolchain).

### Tests development

Every product change should be accompanied with corresponding test modification
(adding new test(s), extending, removing or modifying existing test(s)).

There are 3 types of tests which are used for DPC++ toolchain validation:
* DPC++ in-tree LIT tests including [check-llvm](../../llvm/test),
[check-clang](../../clang/test), [check-llvm-spirv](../../llvm-spirv/test) and
[check-sycl](../../sycl/test) targets stored in this repository. These tests
should not have hardware (e.g. GPU, FPGA, etc.) or external software
dependencies (e.g. OpenCL, Level Zero, CUDA runtimes). All tests not following
this approach should be moved to DPC++ end-to-end or SYCL-CTS tests.
However, the tests for a feature under active development requiring atomic
change for tests and product can be put to
[sycl/test/on-device](../../sycl/test/on-device) temporarily. It is developer
responsibility to move the tests to DPC++ E2E test suite or SYCL-CTS once
the feature is stabilized.

    **Guidelines for adding DPC++ in-tree LIT tests (DPC++ Clang FE tests)**:
    - Use `sycl::` namespace instead of `cl::sycl::`

    - Include sycl mock headers as system headers.
    Example: `-internal-isystem %S/Inputs`
    `#include "sycl.hpp"`

    - Use SYCL functions for invoking kernels from the mock header `(single_task, parallel_for, parallel_for_work_group)`
    Example:
    ```bash
    `#include "Inputs/sycl.hpp"`
    sycl::queue q;
    q.submit([&](cl::sycl::handler &h) {
    h.single_task( { //code });
    });
    ```

    - Add a helpful comment describing what the test does at the beginning and other comments throughout the test as necessary.

    - Try to follow descriptive naming convention for variables, functions as much as possible.
    Please refer [LLVM naming convention](https://llvm.org/docs/CodingStandards.html#name-types-functions-variables-and-enumerators-properly)

* DPC++ end-to-end (E2E) tests which are extension to
[LLVM\* test suite](https://github.com/intel/llvm-test-suite/tree/intel/SYCL).
A test which requires full stack including backend runtimes (e.g. OpenCL,
Level Zero or CUDA) should be put to DPC++ E2E test suite following
[CONTRIBUTING](https://github.com/intel/llvm-test-suite/blob/intel/CONTRIBUTING.md).

* SYCL-CTS are official 
[Khronos\* SYCL\* conformance tests](https://github.com/KhronosGroup/SYCL-CTS).
They verify SYCL specification compatibility. All implementation details or
extensions are out of scope for the tests. If SYCL specification has changed
(SYCL CTS tests conflict with recent version of SYCL specification) or change
is required in the way the tests are built with DPC++ compiler (defined in
[FindIntel_SYCL](https://github.com/KhronosGroup/SYCL-CTS/blob/SYCL-1.2.1/master/cmake/FindIntel_SYCL.cmake))
pull request should be created under
[KhronosGroup/SYCL-CTS](https://github.com/KhronosGroup/SYCL-CTS) with required
patch.

### Commit message

- When writing your commit message, please make sure to follow
  [LLVM developer policies](
  https://llvm.org/docs/DeveloperPolicy.html#commit-messages) on the subject.
- For any DPC++-related commit, the `[SYCL]` tag should be present in the
  commit message title. To a reasonable extent, additional tags can be used
  to signify the component changed, e.g.: `[PI]`, `[CUDA]`, `[Doc]`.
- For product changes which require modification in tests outside of the current repository
  (see [Test DPC++ toolchain](sycl/doc/GetStartedGuide.md#test-dpc-toolchain)),
  the commit message should contain the link to corresponding test PR, e.g.: E2E
  test changes are available under intel/llvm-test-suite#88 or SYCL
  conformance test changes are available under KhronosGroup/SYCL-CTS#65 (see
  [Autolinked references and URLs](https://docs.github.com/en/free-pro-team/github/writing-on-github/autolinked-references-and-urls)
  for more details). The same message should be present both in commit
  message and in PR description.

### Review and acceptance testing

- Create a pull request for your changes following [Creating a pull request
instructions](https://help.github.com/articles/creating-a-pull-request/).
- CI will run a signed-off check as soon as your PR is created - see the
**check_pr** CI action results.
- CI will run several build and functional testing checks as soon as the PR is
approved by an Intel representative.
  - A new approval is needed if the PR was updated (e.g. during code review).
- Once the PR is approved and all checks have passed, the pull request is
ready for merge.

### Merge

Project maintainers merge pull requests using one of the following options:

- [Rebase and merge] The preferable choice for PRs containing a single commit
- [Squash and merge] Used when there are multiple commits in the PR
  - Squashing is done to make sure that the project is buildable on any commit
- [Create a merge commit] Used for LLVM pull-down PRs to preserve hashes of the
commits pulled from the LLVM community repository

*Other names and brands may be claimed as the property of others.
