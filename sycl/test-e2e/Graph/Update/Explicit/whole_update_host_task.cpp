// Sporadically fails on:
// UNSUPPORTED: linux && level_zero && arch-intel_gpu_pvc
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22555

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero && !system-windows %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: aspect-usm_shared_allocations

#define GRAPH_E2E_EXPLICIT

#include "../../Inputs/whole_update_host_task.cpp"
