// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if (level_zero && linux) %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
// RUN: %if (level_zero && windows) %{env UR_L0_LEAKS_DEBUG=1 env SYCL_ENABLE_DEFAULT_CONTEXTS=0 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/multiple_exec_graphs.cpp"
