// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK
//
// Temporarily disabled until failure is addressed.
// UNSUPPORTED: windows

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/add_nodes_after_finalize.cpp"
