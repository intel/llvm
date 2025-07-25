// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}
//
//
// Intended - OpenCL fail due to backend
// The OpenCL emulation layer does not return `CL_INVALID_WORK_GROUP_SIZE` as it
// should. So the Sycl graph support cannot correctly catch the error and throw
// the approriate exception for negative test. An issue has been reported
// https://github.com/bashbaug/SimpleOpenCLSamples/issues/95
// UNSUPPORTED: opencl

#define GRAPH_E2E_RECORD_REPLAY

#include "../Inputs/work_group_size_prop.cpp"
