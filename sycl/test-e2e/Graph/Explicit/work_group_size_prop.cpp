// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if ext_oneapi_level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Note: failing negative test with CUDA & HIP in the original test
// UNSUPPORTED: cuda, hip

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/work_group_size_prop.cpp"
