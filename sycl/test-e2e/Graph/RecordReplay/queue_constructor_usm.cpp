// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Test is flaky on Windows, disable until it can be fixed
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/11852

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/queue_constructor_usm.cpp"
