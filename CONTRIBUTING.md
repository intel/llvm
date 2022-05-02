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

For any changes not related to Intel Project for LLVM\* technology, but rather
to LLVM in general, it is strongly encouraged that you submit the patch to
https://llvm.org/ directly.  See
[LLVM contribution guidelines](https://llvm.org/docs/Contributing.html) for
more information.

Every change should be accompanied with corresponding test modification (adding
new test(s), extending, removing or modifying existing test(s)).

To contribute:

- Create a personal fork of the project on GitHub
- Prepare your patch
  - follow [LLVM coding standards](https://llvm.org/docs/CodingStandards.html)
  - [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and
    [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) tools can be
    integrated into your workflow to ensure formatting and stylistic
    compliance of your changes. To avoid code formatting misalignment with
    GitHub Actions check we recommend using 10 version of clang-format tool
    (default version on Ubuntu 20.04).
  - use

    ```bash
    ./clang/tools/clang-format/git-clang-format `git merge-base origin/sycl HEAD`
    ```

    to check the format of your current changes. `origin/sycl` branch is an
    example here
    - `-f` to also correct unstaged changes
    - `--diff` to only print the diff without applying

#### Project-specific contribution guidelines

- [Contribute to DPC++](/../sycl/sycl/doc/developer/ContributeToDPCPP.md)

### Pull request

- When creating your commit messages, please make sure to follow
  [LLVM developer policies](
  https://llvm.org/docs/DeveloperPolicy.html#commit-messages) on the subject.
  - [The seven rules of a great Git commit message](https://cbea.ms/git-commit)
    are recommended read and follow.
- To a reasonable extent, title tags can be used to signify the component
  changed, e.g.: `[PI]`, `[CUDA]`, `[Doc]`.
- Create a pull request (PR) for your changes following
  [Creating a pull request instructions](https://help.github.com/articles/creating-a-pull-request/).
  - Make sure PR has a good description explaining all of the changes made,
    represented by commits in the PR.
  - When PR is merged all commits are squashed and PR description is used as
    the merged commit message.
  - Consider splitting the large set of changes on small independent PRs that
    can be reviewed, tested and merged independently.
- For changes which require modification in tests outside of the current repository
  the commit message should contain the link to corresponding test PR.
  For example: intel/llvm-test-suite#88 or KhronosGroup/SYCL-CTS#65. (see
  [Autolinked references and URLs](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls)
  for more details). The same message should be present both in commit
  message and in PR description.

### Review and acceptance testing

- CI will run several build and functional testing checks as soon as the PR is
approved by an Intel representative.
  - A new approval is needed if the PR was updated (e.g. during code review).
- Once the PR is approved and all checks have passed, the pull request is
ready for merge.

### Merge

Project gatekeepers (@intel/llvm-gatekeepers) merge pull requests using [Squash
and merge] and using PR description as the commit message, replacing all
individual comments made per commit. Authors of the change must ensure PR
description is up to date at the merge stage, as sometimes comments addressed
during code reviews can invalidate original PR description. Feel free to ping
@intel/llvm-gatekeepers if your PR is green and can be merged.

Pulldown from LLVM upstream is done through merge commits to preserve hashes of
the original commits pulled from the LLVM community repository.

<sub>\*Other names and brands may be claimed as the property of others.</sub>
