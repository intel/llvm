# Contribution guidelines

## If you have found a bug or would like to see a new feature

Please reach us by creating a [new issue].

Your bug report should include a proper description and steps to reproduce:
- attach the LLVM BC or SPV file you are trying to translate and the command you
  launch
- any backtrace in case of crashes would be helpful
- please describe what goes wrong or what is unexpected during translation

For feature requests, please describe the feature you would like to see
implemented in the translator.

[new issue]: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/new

## If you would like to contribute your change

Please open a [pull request]. If you are not sure whether your changes are
correct, you can either mark it as [draft] or create an issue to discuss the
problem and possible ways to fix it prior to publishing a PR.

It is okay to have several commits in the PR, but each of them should be
buildable and tests should pass. Maintainers can squash several commits
into a single one for you during merge, but if you would like to see several
commits in the git history, please let us know in PR description/comments so
maintainers will rebase your PR instead of squashing it.

Each functional change (new feature or bug fix) must be supplied with
corresponding tests. See [#testing-guidelines] for more information about
testing. NFC (non-functional change) PRs can be accepted without new tests.

Code changes should follow coding standards, which are inherited from [LLVM
Coding Standards]. Compliance of your code is checked automatically using
Travis CI. See [clang-format] and [clang-tidy] configs for more details about
coding standards.

## How to add an extension

First of all please make sure you have added a link to the
specification for the extension in your PR. Then to add definitions of
new Op Codes you shall modify [spirv.hpp], which is an external
dependency for this project. To do so, you should add new definitions
to [json grammar file], rebuild the header following the
[instructions] in [SPIR-V Headers repository] and push your changes
for review, i.e. make a PR. Once the PR is merged, a new spirv.hpp
will have to be downloaded during build of the translator; make sure
to update the hash for SPIRV-Headers in [spirv-headers-tag.conf]
so that tokens from your extension can be visible to the translator
build.

It's highly recommended to add the definitions to [SPIR-V Headers repository]
first, but if you don't want to bring it there yet, you can define new Op Codes
in the [internal SPIR-V header file].

For local testing you can copy your spirv.hpp variant to
`<PATH_TO_SPIRV_HEADERS>/include/spirv/unified1` and/or modify it
there. See [README.md](README.md#configuring-spir-v-headers) for build
instructions that should be employed with such modifications.

### Conditions to merge a PR

In order to get your PR merged, the following conditions must be met:
- If you are a first-time contributor, you have to sign the
  [Contributor License Agreement]. Corresponding link and instructions will be
  automatically posted into your PR.
- [Travis CI testing] jobs must pass on your PR: this includes functional
  testing and checking for complying with coding standards.
- You need to get approval from at least one contributor with merge rights.

As a contributor, you should expect that even an approved PR might still be left
open for a few days: this is needed, because the translator is being developed
by different vendors and individuals and we need to ensure that each interested
party is able to react to new changes and provide feedback.

Information below is a guideline for repo maintainers and can be used by
contributors to get some expectations about how long a PR has to be open before
it can be merged:
- For any significant change/redesign, the PR must be open for at least 5
  working days, so everyone interested can step in to provide feedback, discuss
  direction and help to find bugs.
  - Ideally, there should be approvals from different vendors/individuals to get
    it merged, particularly for larger changes.
- For regular changes/bug fixes, the PR must be open for at least 2-3 working
  days, so everyone interested can step in for review and provide feedback.
  - If the change is vendor-specific (bug fix in vendor extension implementation
    or new vendor-specific extension support), then it is okay to merge PR
    sooner.
  - If the change affects or might affect several interested parties, the PR
    must be left open for 2-3 working days and it would be good to see feedback
    from different vendors/inviduals before merging.
- Tiny NFC changes or trivial build fixes (due to LLVM API changes) can be
  submitted as soon as testing is finished and PR approved - no need to wait for
  too long.
- In general, just use common sense to wait long enough to get feedback from
  everyone who might be interested in the PR and don't hesitate to explicitly
  mention individuals who might be interested in reviewing the PR.

[pull request]: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pulls
[draft]: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests
[LLVM Coding Standards]: https://llvm.org/docs/CodingStandards.html
[clang-format]: [.clang-format]
[clang-tidy]: [.clang-tidy]
[spirv.hpp]: https://github.com/KhronosGroup/SPIRV-Headers/blob/master/include/spirv/unified1/spirv.hpp
[json grammar file]: https://github.com/KhronosGroup/SPIRV-Headers/blob/master/include/spirv/unified1/spirv.core.grammar.json
[instructions]: https://github.com/KhronosGroup/SPIRV-Headers#generating-headers-from-the-json-grammar-for-the-spir-v-core-instruction-set
[SPIR-V Headers repository]: https://github.com/KhronosGroup/SPIRV-Headers
[internal SPIR-V header file]: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/lib/SPIRV/libSPIRV/spirv_internal.hpp
[Contributor License Agreement]: https://cla-assistant.io/KhronosGroup/SPIRV-LLVM-Translator
[Travis CI testing]: https://travis-ci.org/KhronosGroup/SPIRV-LLVM-Translator
