// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK
//
// TODO enable cuda once buffer issue investigated and fixed
// UNSUPPORTED: cuda

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/buffer_copy_2d.cpp"
