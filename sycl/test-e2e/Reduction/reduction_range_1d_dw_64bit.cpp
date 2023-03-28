// RUN: %clangxx -DENABLE_64_BIT=true -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// TODO: accelerator may not suport atomics required by the current
// implementation. Once fixed, enable 64bit tests in the main test and remove
// this file
// RUNx: %ACC_RUN_PLACEHOLDER %t.out

#include "reduction_range_1d_dw.cpp"
