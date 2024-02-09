// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --implicit-check-not=LEAK %s %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// USM memset command not supported for OpenCL
// Post-commit test failed
// https://github.com/intel/llvm/actions/runs/7814201804/job/21315560479
// Temporarily disable USM based tests while investigating the bug.
// UNSUPPORTED: opencl, gpu-intel-dg2

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/usm_memset.cpp"
