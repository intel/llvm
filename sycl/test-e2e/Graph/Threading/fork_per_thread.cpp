// RUN: %{build} %threads_lib -o %t.out
// RUN: %{run} %t.out in_order
// RUN: %{run} %t.out out_of_order
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out in_order 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out out_of_order 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/threading_fork_per_thread.cpp"
