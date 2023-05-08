// RUN: %{build} -DENABLE_64_BIT=true -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %{run} %t.out
// TODO: accelerator may not suport atomics required by the current
// implementation. Once fixed, enable 64bit tests in the main test and remove
// this file
// UNSUPPORTED: accelerator

#include "reduction_range_1d_dw.cpp"
