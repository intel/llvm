// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero  %{ %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK
//
// TODO enable cuda once buffer issue investigated and fixed
// UNSUPPORTED: cuda
//
// Device to host copy command not supported for OpenCL
// UNSUPPORTED: opencl

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/buffer_copy_target2host_2d.cpp"
