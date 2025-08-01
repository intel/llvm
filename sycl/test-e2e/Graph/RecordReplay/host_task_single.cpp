// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// Immediate command-list testing is disabled on Windows due to a
// non-deterministic leak of the Level Zero context, and is intended
// to be re-enabled once this can be investigated and fixed.
// https://github.com/intel/llvm/issues/14473

// REQUIRES: aspect-usm_host_allocations

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/host_task_single.cpp"
