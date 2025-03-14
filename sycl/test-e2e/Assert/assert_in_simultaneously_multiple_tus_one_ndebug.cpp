// FIXME flaky fail on CUDA
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/7634
//
// XFAIL: (opencl && gpu)
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/11364
//
// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl %{sycl_target_opts} -DDEFINE_NDEBUG_INFILE2 -I %S/Inputs %S/assert_in_simultaneously_multiple_tus.cpp %S/Inputs/kernels_in_file2.cpp -o %t.out %threads_lib
// RUN: %if cpu %{ %{run} %t.out &> %t.cpu.txt ; FileCheck %s --input-file %t.cpu.txt %}
//
// Since this is a multi-threaded application enable memory tracking and
// deferred release feature in the Level Zero adapter to avoid releasing memory
// too early. This is necessary because currently SYCL RT sets indirect access
// flag for all kernels and the Level Zero runtime doesn't support deferred
// release yet.
// Suppress runtime from printing out error messages, so that the test can
// match on assert message generated by the toolchains.

// DEFINE: %{gpu_env} = env SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY=1 SYCL_PI_SUPPRESS_ERROR_MESSAGE=1
// RUN: %if gpu %{ %{gpu_env} %{run} %t.out &> %t.gpu.txt ; FileCheck %s --input-file %t.gpu.txt %}

// Shouldn't fail on ACC as fallback assert isn't enqueued there
// RUN: %if acc %{ %{run} %t.out &> %t.acc.txt ; FileCheck %s --check-prefix=CHECK-ACC --input-file %t.acc.txt %}
//
// CHECK:      this message from file1
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.
//
// CHECK-ACC: The test ended.
