// RUN: %{build} -o %t.out
// RUN: %if level_zero %{%{run} %t.out %S/../Inputs/Kernels/kernels.spv %}

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/kernel_bundle_spirv.cpp"
