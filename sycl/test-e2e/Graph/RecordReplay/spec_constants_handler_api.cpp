// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
// Extra run to check for immediate-command-list in Level Zero
// RUN: %if level_zero && linux %{env SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// The following limitation is not restricted to Sycl-Graph
// but comes from the orignal test : `SpecConstants/2020/handler-api.cpp`
// FIXME: ACC devices use emulation path, which is not yet supported
// UNSUPPORTED: accelerator

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/spec_constants_handler_api.cpp"
