// XFAIL: run-mode && linux && arch-intel_gpu_bmg_g21 && spirv-backend
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19586
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#define GRAPH_E2E_EXPLICIT

#include "../../Inputs/whole_update_work_group_memory.cpp"
