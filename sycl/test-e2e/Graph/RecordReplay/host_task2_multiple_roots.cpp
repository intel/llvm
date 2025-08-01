// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: aspect-usm_shared_allocations

// Intended - Concurrent access to shared USM allocations is not supported by
// CUDA on Windows
// UNSUPPORTED: cuda && windows

// Test is flaky on Windows for all targets, disable until it can be fixed
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/11852

// UNSUPPORTED: level_zero_v2_adapter && arch-intel_gpu_bmg_g21
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/18996

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/host_task2_multiple_roots.cpp"
