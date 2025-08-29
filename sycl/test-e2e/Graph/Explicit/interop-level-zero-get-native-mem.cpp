// REQUIRES: level_zero, level_zero_dev_kit
// UNSUPPORTED: ze_debug
// UNSUPPORTED-INTENDED: Leaks detection is done at UR level and doesn't account
// for native L0 API calls.
// UNSUPPORTED: linux && (gpu-intel-dg2 || arch-intel_gpu_bmg_g21) && run-mode && !igc-dev
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/18273
// XFAIL: arch-intel_gpu_ptl_u || arch-intel_gpu_ptl_h
// XFAIL-TRACKER: CMPLRTST-27745
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/interop-level-zero-get-native-mem.cpp"
