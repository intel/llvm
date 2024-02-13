// REQUIRES: level_zero, level_zero_dev_kit
// L0 plugin incorrectly reports memory leaks because it doesn't take into
// account direct calls to L0 API.
// UNSUPPORTED: ze_debug
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{run} %t.out
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 %{run} %t.out %}

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/interop-level-zero-get-native-mem.cpp"
