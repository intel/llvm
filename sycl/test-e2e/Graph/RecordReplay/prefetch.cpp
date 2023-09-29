// REQUIRES: cuda || level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=2 %{run} %t.out

// CHECK: piextCommandBufferPrefetchUSM

// Since Prefetch is only a memory hint that doesn't
// impact results but only performances, we verify
// that a node is correctly added by checking PI function calls

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/prefetch.cpp"
