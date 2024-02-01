// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// XUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --implicit-check-not=LEAK %s %}
// RUN: %if (level_zero && linux) %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck --implicit-check-not=LEAK %s %}
// RUN: %if (level_zero && windows) %{env UR_L0_LEAKS_DEBUG=1 env SYCL_ENABLE_DEFAULT_CONTEXTS=0 %{run} %t.out 2>&1 | FileCheck --implicit-check-not=LEAK %s %}

// USM memset command not supported for OpenCL
// UNSUPPORTED: opencl

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/usm_memset.cpp"
