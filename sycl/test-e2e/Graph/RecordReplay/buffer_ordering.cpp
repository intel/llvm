// REQUIRES: level_zero, gpu

// https://github.com/intel/llvm/issues/7585 to fix the flaky failure:
// UNSUPPORTED: level_zero

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using ZE_DEBUG
// RUN: %if ext_oneapi_level_zero %{env ZE_DEBUG=4 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/buffer_ordering.cpp"
