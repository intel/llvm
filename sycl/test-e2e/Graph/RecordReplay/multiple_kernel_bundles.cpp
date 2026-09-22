// UNSUPPORTED: aot-only
// UNSUPPORTED-INTENDED: get_kernel_bundle<bundle_state::input> needs a JIT-capable image.

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/multiple_kernel_bundles.cpp"
