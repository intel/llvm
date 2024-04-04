## Overview

The "Graph" directory contains tests for the
[sycl_ext_oneapi_graph](../../doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc)
extension.

Each subdirectory contains a `lit.local.cfg` file. This file sets the `lit`
option `config.required_features` to the graph aspect required to run the tests:

- `aspect-ext_oneapi_limited_graph` for any test that doesn't require the
  executable graph update feature.
- `aspect-ext_oneapi_graph` for any test that does require the executable graph
  update feature.

## Structure

Most of the tests are written in a similar manner to other `e2e` tests. The
exception to this are tests in the `Inputs` directory which are meant to be used
as inputs to other tests.

Often, the same feature, needs to be tested for both the `Explicit`
and `Record and Replay` APIs. To avoid code duplication, such tests are added to
the `Inputs` folder and rely on common code from `graph_common.hpp` to construct
the graph. The files in the `Inputs` directory are not run directly by `lit`.
Instead, that source is included by tests in the `Explicit` and `RecordReplay`
directories. These tests also define `GRAPH_E2E_EXPLICIT`
or `GRAPH_E2E_RECORD_REPLAY` respectively to choose which API is used by the
common code.

The other directories are used to group similar tests together. They might
themselves contain subdirectories named `Explicit` and `RecordReplay` if they
make use of the framework described above.

In addition, in order to help identify specific tests, the matching files
in `Explicit`, `RecordReplay` and `Inputs` folders should have identical names.
