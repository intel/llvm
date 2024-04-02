# Overview

The "Graph" directory contains tests for the sycl_ext_oneapi_graph extension.

# Structure

Most of the tests are written in a similar manner to other e2e tests. The
exception to this are tests in the `Inputs` folder which are meant to be used
as inputs to other tests.

Often, the same feature, needs to be tested for both the `Explicit`
and `Record and Replay` APIs. To avoid code duplication, such tests are added to
the `Inputs` folder and rely on common code from `graph_common.hpp` to construct
the graph. The files in the `Inputs` folder are not run directly by `lit`.
Instead, that source is included by tests in the `Explicit` and `RecordReplay`
folders. These tests also define `GRAPH_E2E_EXPLICIT`
or `GRAPH_E2E_RECORD_REPLAY` respectively to choose which API is used by the
common code.

In addition, in order to help identify specific tests, the matching files
in `Explicit`, `RecordReplay` and `Inputs` folders should have identical names.
