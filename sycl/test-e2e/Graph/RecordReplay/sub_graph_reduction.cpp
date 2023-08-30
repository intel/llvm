// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Expected fail as reduction support is not complete.
// XFAIL: *

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/sub_graph_reduction.cpp"
