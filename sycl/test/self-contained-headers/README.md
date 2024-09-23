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

## Implementation

There was a couple of iterations on the suite design and its current shape
features the following:
- each header in `build/include/sycl` is checked as a separate test
- each such test is generated on the fly dynamically during LIT discovery phase

That is done to allow for massive parallelism and keep those tests small and
quick.

Absolute most of the magic is happenning within
[`sycl/test/format.py`](/sycl/test/format.py): we define a custom test format in
there which overrides standard discovery and test execution rules.

## How to use and maintain

Those tests are part of `check-sycl` target and you can pass a regexp acepted
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

Since there are no dedicated files for each test, `XFAIL`ing them using regular
method is impossible, but it is still supported. To do so, open
[the local config](/sycl/test/self-contained-headers/lit.local.cfg) and modify
list of files which should be treated as expected to fail.

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

### Old legacy files in build/ area are still checked

The custom discovery script uses `build/include/sycl/` folder contents to
generate tests for each header it finds there. It means that if some header was
removed from the codebase, it may still be present in `build` folder unless
some cleanup is performed.

### No OS-specific `XFAIL` mechanism is implemented

`XFAIL` mechanism mentioned in "How to use and maintain" section does not
support marking a test as expected to fail only in certain environment, which
may be problematic for headers which trigger some differences between different
OS-es, for example.
