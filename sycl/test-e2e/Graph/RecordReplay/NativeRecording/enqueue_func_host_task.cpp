// REQUIRES: level_zero_v2_adapter
// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: arch-intel_gpu_bmg_g21

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests that syclex::host_task() can be recorded into a native-recording SYCL
// Graph and executes correctly between two SYCL kernels.

#define GRAPH_E2E_NATIVE_RECORDING

#include "../Inputs/enqueue_func_host_task.cpp"
