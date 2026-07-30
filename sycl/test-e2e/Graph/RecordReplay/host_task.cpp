// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// UNSUPPORTED: windows && level_zero && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h)
// UNSUPPORTED-TRACKER: CMPLRLLVM-74630

// REQUIRES: aspect-usm_shared_allocations

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/host_task.cpp"
