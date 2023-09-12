// REQUIRES: level_zero, gpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 2>&1 | FileCheck %s
//
// CHECK: complete

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/event_status_querying.cpp"
