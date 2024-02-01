// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/host_task2.cpp"
