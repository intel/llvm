// REQUIRES: windows
// REQUIRES: build-and-run-mode

// DEFINE: %{sharedflag} = %if cl_options %{/clang:-shared%} %else %{-shared%}

// RUN: %clangxx %{sharedflag} -fsycl %{sycl_target_opts} %S\Inputs\kernel_module.cpp -o %t.dll
// RUN: %clangxx -DTEST_SHARED_LIB='"%/t.dll"' -fsycl %{sycl_target_opts} %S\Inputs\kernel_function.cpp -o %t.out
// RUN: %{run} %t.out
