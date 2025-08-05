// REQUIRES: linux
// TODO: Supported for ROCM 5. Further development required to support AMDGPU.
// UNSUPPORTED: hip

// RUN: %clangxx -fPIC -shared -fsycl %{sycl_target_opts} %S/Inputs/kernel_module.cpp -o %t.so
// RUN: %clangxx -DTEST_SHARED_LIB='"%t.so"' -ldl -fsycl %{sycl_target_opts} %S/Inputs/kernel_function.cpp -o %t.out
// RUN: %clangxx -DTEST_SHARED_LIB='"kernel_lin.cpp.tmp.so"' -ldl -fsycl %{sycl_target_opts} %S/Inputs/kernel_function.cpp -o %t2.out
// RUN: %{run} %t.out
