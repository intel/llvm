// REQUIRES: level_zero_v2_adapter && (arch-intel_gpu_bmg_g21 || arch-intel_gpu_bmg_g31 || arch-intel_gpu_cri)
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define GRAPH_E2E_RECORD_REPLAY
#define GRAPH_E2E_NATIVE_RECORDING

#include "../../Inputs/destruction_callback.cpp"
