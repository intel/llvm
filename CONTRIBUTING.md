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

### Commit message

- When writing your commit message, please make sure to follow
  [LLVM developer policies](
  https://llvm.org/docs/DeveloperPolicy.html#commit-messages) on the subject.
- For any DPC++-related commit, the `[SYCL]` tag should be present in the
  commit message title. To a reasonable extent, additional tags can be used
  to signify the component changed, e.g.: `[PI]`, `[CUDA]`, `[Doc]`.
- For product changes which require modification of E2E tests
  (see [Test DPC++ toolchain](sycl/doc/GetStartedGuide.md#test-dpc-toolchain))
  the commit message should contain link to corresponding test PR, e.g.: E2E
  test changes are availbale under intel/llvm-test-suite#88.
  See [Autolinked references and URLs](https://docs.github.com/en/free-pro-team/github/writing-on-github/autolinked-references-and-urls)
  for more details.

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
