// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// XFAIL: *
// Subgraph doesn't work properly in second parent graph

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/sub_graph_two_parent_graphs.cpp"