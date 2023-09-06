// REQUIRES: windows

// RUN: %clangxx -shared -fsycl -fsycl-targets=%{sycl_triple} %S\Inputs\kernel_module.cpp -o %t.input
// RUN: %clangxx -DTEST_SHARED_LIB='"%/t.input"' -fsycl -fsycl-targets=%{sycl_triple} %S\Inputs\kernel_function.cpp -o %t.out
// RUN: %{run} %t.out