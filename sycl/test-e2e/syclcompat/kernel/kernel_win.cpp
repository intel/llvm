// REQUIRES: windows

// Currently disabled due to flaky failures caused by Windows runtime not
// unregistering the binaries when runtime-loaded .dll files with SYCL binaries
// are unloaded.
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: CMPLRLLVM-68687

// DEFINE: %{sharedflag} = %if cl_options %{/clang:-shared%} %else %{-shared%}

// This test is sensitive to the absolute path of the dll file produced, so we
// run the test completely on the run system to avoid issues.

// RUN: %{run-aux} %clangxx %{sharedflag} -fsycl %{sycl_target_opts} %S\Inputs\kernel_module.cpp -o %t.dll
// RUN: %{run-aux} %clangxx -DTEST_SHARED_LIB='"%/t.dll"' -fsycl %{sycl_target_opts} %S\Inputs\kernel_function.cpp -o %t.out
// RUN: %{run} %t.out
