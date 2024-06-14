// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -DNDEBUG  %S/assert_in_one_kernel.cpp -o %t.out
// RUN: %{run} %t.out | FileCheck %s

// flaky on OCL GPU, CPU and FPGA.
// Seems to sometimes just terminate everything when piProgramRelease is being called.
// the PI Trace seems right. We create exactly one program and it has
// one call to piProgramRetain followed by two calls to piProgramRelease.
// Yet OpenCL is sometimes crashing on that second call on Windows

// UNSUPPORTED: opencl && windows

//
// CHECK-NOT: from assert statement
// CHECK: The test ended.
