// REQUIRES: aspect-usm_shared_allocations

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests fork/join of a restricted host_task using the handler path.

#include "Inputs/enqueue_func_host_task_event.cpp"
