// REQUIRES: level_zero_v2_adapter && arch-intel_gpu_bmg_g21

// Checks that info::graph::recording_mode reports effective state rather than
// property-list contents.

// RUN: %{build} -o %t.out
// RUN: env SYCL_GRAPH_FORCE_NATIVE_RECORDING=1 %{run} %t.out
// RUN: %if level_zero %{env SYCL_GRAPH_FORCE_NATIVE_RECORDING=1 %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

#define GRAPH_E2E_RECORD_REPLAY
#define GRAPH_E2E_FORCE_NATIVE_RECORDING_ENV

#include "../../Inputs/query_info.cpp"
