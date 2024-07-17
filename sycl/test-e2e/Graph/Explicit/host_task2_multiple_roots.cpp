// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: aspect-usm_shared_allocations

// Intended - Concurrent access to shared USM allocations is not supported by
// CUDA on Windows
// UNSUPPORTED: cuda && windows

// Test is flaky on Windows for all targets, disable until it can be fixed
// https://github.com/intel/llvm/issues/11852
// UNSUPPORTED: windows

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/host_task2_multiple_roots.cpp"
