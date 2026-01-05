# Releases at intel/llvm

intel/llvm provides a few ways for you to get SYCL compiler and this
page is intended to document them as well as provide links and instructions
how to you use them.

## Nightly builds

The project is being built and tested every night and those builds
are published on GitHub as `nightly-YYYY-MM-DD` tags.

Those builds are of **pre-release** quality: they are only suitable
for early integration and testing of a bleeding-edge compiler.

Because those builds are made from the latest commit available,
there is no guarantee of any stability of those builds.

Please note that those builds are tailored for our own testing needs
and therefore they are built with assertions enabled and may have additional
dependencies.

## Official releases

Official release branches are occasionally created to provide
stable and validated commits for end users. The list below is
ordered from the newest to the oldest.

Release branches are prefixed with `sycl-rel-`

### _[Upcoming]_ `6.3.0` release

This release will be made from the
[`sycl-rel-6_3`](https://github.com/intel/llvm/tree/sycl-rel-6_3) branch and we
have the following tags published:
- [`v6.3.0-rc1`](https://github.com/intel/llvm/releases/tag/v6.3.0-rc1) -
  release candidate 1 for 6.3.0 release

### **[Latest]** `6.2.0` release

This release was made from
the [`sycl-rel-6_2`](https://github.com/intel/llvm/tree/sycl-rel-6_2) branch and
we have the following tags published:
- **[Latest]** [`v6.2.0`](https://github.com/intel/llvm/releases/tag/v6.2.0)
- [`v6.2.0-rc1`](https://github.com/intel/llvm/releases/tag/v6.2.0-rc1) -
  release candidate 1 for 6.2.0 release

### [Legacy] releases

These releases are _legacy_ meaning that we do not provide support for the
corresponding release branches anymore, no bug fixes (functional or security)
will be backported to them and no new tags will be made.

#### `6.1.0` release

This release was made from
the [`sycl-rel-6_1_0`](https://github.com/intel/llvm/tree/sycl-rel-6_1_0)
branch and we have the following tags published:
- [`v6.1.0`](https://github.com/intel/llvm/releases/tag/v6.1.0)

#### `6.0.X` releases

These releases are made from
the [`sycl-rel-6_0_0`](https://github.com/intel/llvm/tree/sycl-rel-6_0_0)
branch and we have the following tags published:
- [`v6.0.1`](https://github.com/intel/llvm/releases/tag/v6.0.1) -
  6.0.1 bugfix release
- [`v6.0.0`](https://github.com/intel/llvm/releases/tag/v6.0.0) -
  6.0.0 release
- [`v6.0.0-rc1`](https://github.com/intel/llvm/releases/tag/v6.0.0-rc1) -
  release candidate 1 for 6.0.0 release
