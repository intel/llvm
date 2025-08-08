# Working on a release branch

A "release branch" is defined as a branch whose name starts with `sycl-rel-`
prefix.

Those branches are intended to indicate stable snapshots of our product so that
our users don't need to guess which nightly build is good enough for their
needs.

Therefore, those branches have higher quality requirements and as such have
different contribution rules intended to preserve their stability.

If you are not familiar with the [general contribution guidelines][contributing]
or the [DPC++ specific contribution guidelines][contributing-to-dpcpp], please
familiarize yourself with those documents first because they also apply to
release branches.

## Extra rules for release branches

### Only cherry-picks are allowed

It is assumed that everything you do on a release branch should also be
repeated on the default `sycl` branch to ensure that it is automatically
included into future releases.

Therefore, when submitting a PR to a release branch, its description should
contain a link to the corresponding PR in the default `sycl` branch.

Note that it is not acceptable to first merge something into a
release branch and then apply it to the default `sycl` branch. The flow goes in
the opposite direction where you first land a patch to the default `sycl` branch
and then backport it to a release branch.

### No new features are allowed

Features are generally more complicated than bug fixes and may require further
bug fixes as well. Considering that release branches are intended to be stable,
no new features are allowed to be added there.

[contributing]: https://github.com/intel/llvm/blob/sycl/CONTRIBUTING.md
[contributing-to-dpcpp]: ./ContributeToDPCPP.md
