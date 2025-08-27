// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
// UNSUPPORTED: target-amd
// UNSUPPORTED-INTENDED: sycl_ext_oneapi_work_group_static is not supported on
// AMD

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/work_group_static_memory.cpp"
