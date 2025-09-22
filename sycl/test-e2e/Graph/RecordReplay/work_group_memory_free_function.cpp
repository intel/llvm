// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// XFAIL: spirv-backend
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/18230

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/work_group_memory_free_function.cpp"
