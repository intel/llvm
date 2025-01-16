// REQUIRES: linux
// TODO: Supported for ROCM 5. Further development required to support AMDGPU.
// UNSUPPORTED: target-amd

// RUN: %clangxx -fPIC -shared -fsycl %{sycl_target_opts} %S/Inputs/kernel_module.cpp -o %t.so
// RUN: %clangxx -DTEST_SHARED_LIB='"%t.so"' -ldl -fsycl %{sycl_target_opts} %S/Inputs/kernel_function.cpp -o %t.out
// RUN: %{run} %t.out
