// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG.
// The leak check is restricted to the v2 adapter: on the v1 adapter it
// sporadically reports a leak (https://github.com/intel/llvm/issues/22555).
// RUN: %if level_zero_v2_adapter && !system-windows %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// REQUIRES: aspect-usm_shared_allocations

#define GRAPH_E2E_RECORD_REPLAY

#include "../../Inputs/whole_update_host_task.cpp"
