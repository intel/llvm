# Releases at intel/llvm

intel/llvm provides a few ways for you to get SYCL compiler and this
page is intended to document them, provide links and instructions
how to you use them.

## Nightly builds

The project is being build and tested every night and those builds
are published on GitHub as `nightly-YYYY-MM-DD` tags.

Those builds are of **pre-release** quality: they are only suitable
for early integration and testing of a bleeding edge compiler.

Because those builds are made from the latest commit available,
there is no guarantee of any stability of those builds.

Please note that those builds are tailored for our own testing needs
and therefore they build with assertions enabled and have some extra
dependencies.

For example, for extra testing we link with
[`KhronosGroup/SPIRV-Tools`](https://github.com/KhronosGroup/SPIRV-Tools)
and you will need to make
`SPIRV-Tools-shared.dll`/`libSPIRV-Tools-shared.so` available in
your `PATH`/`LD_LIBRARY_PATH`.

## Official releases

Official release branches are created from time to time to provide
stable and validated commits for end users. The list below is
ordered from the newest to the oldest.

Release branches are prefixed with `sycl-rel-`

### [Upcoming] `6.1.0` release

This release will be made from
the [`sycl-rel-6_1_0`](https://github.com/intel/llvm/tree/sycl-rel-6_1_0)

At this moment, no tags are created on the branch, it is being
stabilized.

### `6.0.X` releases

Theses releases are made from
the [`sycl-rel-6_0_0`](https://github.com/intel/llvm/tree/sycl-rel-6_0_0)
and we have the following tags published:
- [`v6.0.1`](https://github.com/intel/llvm/releases/tag/v6.0.1) -
  6.0.1 bugfix release
- [`v6.0.0`](https://github.com/intel/llvm/releases/tag/v6.0.0) -
  6.0.0 release
- [`v6.0.0-rc1`](https://github.com/intel/llvm/releases/tag/v6.0.0-rc1) -
  release candidate 1 for 6.0.0 release
