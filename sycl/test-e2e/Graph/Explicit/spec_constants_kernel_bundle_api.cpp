// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

// Intended - The following limitation is not restricted to Sycl-Graph
// but comes from the orignal test : `SpecConstants/2020/kernel-bundle-api.cpp`
// UNSUPPORTED: hip

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/spec_constants_kernel_bundle_api.cpp"
