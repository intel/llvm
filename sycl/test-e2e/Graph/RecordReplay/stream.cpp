// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out %GPU_CHECK_PLACEHOLDER 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Expected fail as sycl::stream is not implemented yet
// XFAIL: *

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/stream.cpp"