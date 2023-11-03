// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %GPU_CHECK_PLACEHOLDER
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out %GPU_CHECK_PLACEHOLDER 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Expected fail as sycl streams aren't implemented yet
// XFAIL: *

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/stream.cpp"

// CHECK-DAG: Val: 1
// CHECK-DAG: Val: 2
// CHECK-DAG: Val: 3
// CHECK-DAG: Val: 4
// CHECK-DAG: Val: 5
// CHECK-DAG: Val: 6
// CHECK-DAG: Val: 7
// CHECK-DAG: Val: 8
// CHECK-DAG: Val: 9
// CHECK-DAG: Val: 10
// CHECK-DAG: Val: 11
// CHECK-DAG: Val: 12
// CHECK-DAG: Val: 13
// CHECK-DAG: Val: 14
// CHECK-DAG: Val: 15
// CHECK-DAG: Val: 16
