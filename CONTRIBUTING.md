# Contributing

## License
Intel Project for LLVM* technology is licensed under the terms of the
Apache-2.0 with LLVM-exception license ([LICENSE.txt](llvm/LICENSE.TXT))
to ensure our ability to contribute this project to the LLVM project
under the same license.

By contributing to this project, you agree to the Apache-2.0 with
LLVM-exception license and copyright terms there in and release your
contribution under these terms.

## Sign your work
Please use the sign-off line at the end of your contribution. Your
signature certifies that you wrote the contribution or otherwise have
the right to pass it on as an open-source contribution, and that you
agree to provide your contribution under the terms of the licenses
noted above. The rules are pretty simple: if you can certify the
below (from [developercertificate.org](http://developercertificate.org)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

## Contribution process

### Development

- Create a personal fork of the project on GitHub
- Use **sycl** branch as baseline for your changes
- Prepare your patch (follow
  [LLVM coding standards](https://llvm.org/docs/CodingStandards.html))
- Build the project and run all tests (see
[GetStartedWithSYCLCompiler.md](sycl/doc/GetStartedWithSYCLCompiler.md))

### Review and acceptance testing

- Create a pull request for your changes following [Creating a pull request
instructions](https://help.github.com/articles/creating-a-pull-request/)
- CI will run signed-off check as soon as PR is created, see **check_pr** CI
check for results
- CI will run build and functional testing check as soon as PR is approved by
Intel representative
  - New approval is needed if PR was updated (e.g. during code review)
- Once PR is approved and all checks pass, the pull request is ready for merge

### Merge

Project maintainers merge pull requests using one of the following options:
- [Rebase and merge] The preferable choice for PRs containing a single commit
- [Squash and merge] Used when there are multiple commits in the PR
  - Squashing is done to make sure that the project is buildable on any commit
- [Create a merge commit] Used for LLVM pull-down PRs to preserve hashes of the
commits pulled from the LLVM community repository