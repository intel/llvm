// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK
//
// TODO enable cuda once buffer issue investigated and fixed
// UNSUPPORTED: cuda
//
// Host to device copy command not supported for OpenCL
// UNSUPPORTED: opencl

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/buffer_copy_host2target_offset.cpp"
