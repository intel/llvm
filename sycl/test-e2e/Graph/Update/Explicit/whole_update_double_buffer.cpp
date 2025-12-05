// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// XFAIL: arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h || arch-intel_gpu_wcl
// XFAIL-TRACKER: CMPLRTST-27275
// XFAIL-TRACKER: CMPLRLLVM-72055

#define GRAPH_E2E_EXPLICIT

#include "../../Inputs/whole_update_double_buffer.cpp"
