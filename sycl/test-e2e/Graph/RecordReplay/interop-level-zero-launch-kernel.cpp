// REQUIRES: level_zero, level_zero_dev_kit
// L0 adapter incorrectly reports memory leaks because it doesn't take into
// account direct calls to the L0 API.
// UNSUPPORTED: ze_debug
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out %S/../Inputs/Kernels/saxpy.spv
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/saxpy.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out %S/../Inputs/Kernels/saxpy.spv 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/interop-level-zero-launch-kernel.cpp"
