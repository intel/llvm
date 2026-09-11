// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_LAUNCH_BLOCKING=1 %{run} %t.out

// REQUIRES: aspect-usm_shared_allocations

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/launch_blocking.cpp"
