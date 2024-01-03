// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_LEAKS_DEBUG=1 %{run} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Temporarily disabled for CUDA.
// XFAIL: cuda
// Note: failing negative test with HIP in the original test
// TODO: disable hip when HIP backend will be supported by Graph

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/work_group_size_prop.cpp"
