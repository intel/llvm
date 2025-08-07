// By default functors are no longer decomposed preventing the use of set_arg in
// this test, -fsycl-decompose-functor is used to force the old behavior
//
// RUN: %{build} -fsycl-decompose-functor -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
//
// Temporarily disabled until failure is addressed.
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/11852

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/add_nodes_after_finalize.cpp"
