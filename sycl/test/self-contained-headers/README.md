# Self-contained SYCL headers

## Overview

This subfolder contains a special test suite that is intended to check that our
SYCL headers are self-contained, i.e. if you include a single SYCL header and
compile an empty `.cpp` file it should work just fine.

The intent of having such tests is to provide a mechanism to us to write cleaner
code: with tests like this you can do more aggressive `#include`s cleanup and
still be sure that we haven't accidentally removed a necessary `#include`.

**Note:** headers are checked by the suite are compiled in non-system mode,
meaning that any warnings coming out of them may be turned into errors and will
affect test results. This is considered as an extra feature of the suite.

**One more note:** due to templated nature of SYCL headers, not every code path
may be instantiated by a mere `#include` and therefore not every dependency will
be highlighted by a simple test. To overcome this, there is an ability to write
dedicated tests for certain headers which are more exhaustive than a simple
`#include`, see more details below.

## Implementation

There was a couple of iterations on the suite design and its current shape
features the following:
- each header in `build/include/sycl` is checked as a separate test, unless:
  - it doesn't exists in `source/include/sycl`, meaning that it is likely
    removed from the codebase, but still resides in `build/` directory
  - **TODO:** we also have some auto-generated headers which could be skipped
    this way, we need to consider a mechanism to handle them as well
  - **TODO:** presence of outdated headers in `build` directory should also be
    detected, or otherwise it can lead to compilation issues being hidden in
    local setup
- each such test is generated on the fly dynamically during LIT discovery phase,
  unless:
  - there is a special/dedicated test for a header, more details below

That is done to allow for massive parallelism and keep those tests small and
quick.

Absolute most of the magic is happening within
[`sycl/test/format.py`](/sycl/test/format.py): we define a custom test format in
there which overrides standard discovery and test execution rules.

## How to use and maintain

Those tests are part of `check-sycl` target and you can pass a regexp accepted
by Python's `re` package as `SYCL_HEADERS_FILTER` parameter to LIT to filter
which headers you would like to see checked (only those that match the passed
regexp will be used to generate tests).

```
LIT_FILTER='self-contained-headers' llvm-lit sycl/test -DSYCL_HEADERS_FILTER="ext/.*"
```

Note: `LIT_FILTER` env variable is used here to limit LIT tests discovery to
`self-contained-headers` subfolders. Without this env variable, all other LIT
tests will be launched as well.

Documentation for Python's regexp can be found [here][python-3-re].

[python-3-re]: https://docs.python.org/3/library/re.html#regular-expression-syntax

Since there are no dedicated files for auto-generated tests, `XFAIL`ing them
using regular method is impossible, but it is still supported. To do so, open
[the local config](/sycl/test/self-contained-headers/lit.local.cfg) and modify
list of files which should be treated as expected to fail.

### Special tests

As noted above, to truly ensure that SYCL headers are self-contained, we need
not only include them, but also use them
(read: instantiate all classes and methods).

To support that, for every SYCL header we have in `source/include/sycl` the tool
first checks if there is a corresponding test file in
`source/test/self-contained-headers` and if so, it is used instead of an
auto-generated one.

Those special tests should be named and located in certain place to be detected,
or otherwise they will be ignored. For a header
`source/include/sycl/path/to/header.hpp` its special test should be placed under
`source/test/sycl/self-contained-headers/sycl/path/to/header.hpp.cpp`.

Note a few things: directory structure should exactly match, the filename should
be the same as the header file name, but with `.cpp` extension added on top of
it.

Those special tests will be treated as any other regular Sh-based tests, i.e.
you should write your regular `RUN` lines in there. It is expected that those
tests will run a compilation under `-fsyntax-only` mode and verify absence of
any compilation errors or warnings through `-Xclang -verify` mechanism.

Special tests can be `XFAIL`-ed using a regular LIT mechanism.

## Known issues and quirks

### To launch the suite directly, use `LIT_FILTER` env variable

The following command:

```
llvm-lit sycl/test/self-contained-headers
```

Will results in LIT saying that no tests were discovered.

Instead, the following approach should be used:

```
LIT_FILTER='self-contained-headers' llvm-lit sycl/test
```

### No OS-specific `XFAIL` mechanism is implemented for auto-generated tests

`XFAIL` mechanism mentioned in "How to use and maintain" section does not
support marking a test as expected to fail only in certain environment, which
may be problematic for headers which trigger some differences between different
OS-es, for example.
