// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{env UR_L0_LEAKS_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s %}
//
// CHECK-NOT: LEAK

// Temporarily disabled for CUDA and OpenCL
// The OpenCL emulation layer does not return `CL_INVALID_WORK_GROUP_SIZE` as it
// should. So the Sycl graph support cannot correctly catch the error and throw
// the approriate exception for negative test. An issue has been reported
// https://github.com/bashbaug/SimpleOpenCLSamples/issues/95
// XFAIL: cuda, opencl
// Note: failing negative test with HIP in the original test
// TODO: disable hip when HIP backend will be supported by Graph

#define GRAPH_E2E_EXPLICIT

#include "../Inputs/work_group_size_prop.cpp"
