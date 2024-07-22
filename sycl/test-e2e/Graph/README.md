## Overview

The "Graph" directory contains tests for the
[sycl_ext_oneapi_graph](../../doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)
extension.

Many subdirectories contain a `lit.local.cfg` file. This file sets the `lit`
option `config.required_features` to the graph aspect required to run the tests:

- `aspect-ext_oneapi_limited_graph` for any test that doesn't require the
  executable graph update feature.
- `aspect-ext_oneapi_graph` for any test that does require the executable graph
  update feature.

## Structure

Most of the tests are written in a similar manner to other `e2e` tests. The
exception to this are tests in the `Inputs` directory which are meant to be used
as inputs to other tests. The `Kernels` subdirectory of `Inputs` contains SPIR-V
kernels for testing. These can be generated from SYCL kernels by using the
`-fsycl-dump-device-code=<dir>` option to the DPC++ compiler.

Often, the same feature, needs to be tested for both the `Explicit`
and `Record and Replay` APIs. To avoid code duplication, such tests are added to
the `Inputs` folder and rely on common code from `graph_common.hpp` to construct
the graph. The files in the `Inputs` directory are not run directly by `lit`.
Instead, that source is included by tests in the `Explicit` and `RecordReplay`
directories. These tests also define `GRAPH_E2E_EXPLICIT`
or `GRAPH_E2E_RECORD_REPLAY` respectively to choose which API is used by the
common code.

The other directories are used to group similar tests together. Tests that
require a specific `aspect` are also grouped together in order to use the
`lit.local.cfg` file. Directories might themselves contain subdirectories named
`Explicit` and `RecordReplay` if they make use of the framework described above.

In addition, in order to help identify specific tests, the matching files
in `Explicit`, `RecordReplay` and `Inputs` folders should have identical names.

## Test Execution

Tests might be run multiple times using different options. The most commonly used
options are:

- `l0_leak_check`: `lit` substitution which, on the `level-zero` backend, enables
checks for memory leaks caused by mismatched number of calls to memory
allocation / release APIs in `Unified Runtime`.
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`: Environment variable which, 
on the `level-zero` backend, enables or disables
[immediate command-lists](https://spec.oneapi.io/level-zero/latest/core/PROG.html#low-latency-immediate-command-lists).
Without this option, depending on the hardware, immediate command-lists might be
enabled or disabled by default.

Tests might be run multiple times using different combination of the options 
described above. Most tests do the following:

1. A default run which runs for all backends. 
2. On the `level-zero` backend only, tests for leaks and forcefully **disables**
immediate command-lists in order to test this codepath on hardware that enables
immediate command-lists by default.
3. On the `level-zero` backend only, tests for leaks and forcefully **enables**
immediate command-lists in order to test this codepath on hardware that disables
immediate command-lists by default.
