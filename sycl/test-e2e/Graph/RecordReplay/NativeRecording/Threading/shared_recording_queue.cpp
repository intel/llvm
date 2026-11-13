// REQUIRES: level_zero_v2_adapter && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)

// RUN: %{build} %threads_lib -o %t.out
// RUN: %{run} %t.out in_order
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out in_order 2>&1 | FileCheck %s --implicit-check-not=%{leak_check_not} %}

#define GRAPH_E2E_RECORD_REPLAY
#define GRAPH_E2E_NATIVE_RECORDING

#include "../../../Inputs/threading_shared_recording_queue.cpp"
