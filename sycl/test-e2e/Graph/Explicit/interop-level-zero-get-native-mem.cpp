// REQUIRES: level_zero, level_zero_dev_kit
// UNSUPPORTED: ze_debug
// UNSUPPORTED-INTENDED: Leaks detection is done at UR level and doesn't account
// for native L0 API calls.
// UNSUPPORTED: linux && gpu-intel-dg2 && run-mode && !igc-dev
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/18273
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/interop-level-zero-get-native-mem.cpp"
