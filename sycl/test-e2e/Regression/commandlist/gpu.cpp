// REQUIRES: gpu

// RUN: %clangxx -Wno-error=vla-cxx-extension -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/FindPrimesSYCL.cpp %S/Inputs/main.cpp -o %t.out %threads_lib
// RUN: %{run} %t.out
