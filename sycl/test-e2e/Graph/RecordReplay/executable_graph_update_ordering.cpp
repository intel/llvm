// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Skip as executable graph update and host tasks both aren't
// implemented.
// REQUIRES: NOT_YET_IMPLEMENTED

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/executable_graph_update_ordering.cpp"
