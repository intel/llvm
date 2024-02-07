// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK
//
// Post-commit test failed https://github.com/intel/llvm/actions/runs/7814201804/job/21315560479
// Temporary disable the tests while investigating the bug.
// UNSUPPORTED: gpu-intel-dg2

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/queue_constructor_usm.cpp"
