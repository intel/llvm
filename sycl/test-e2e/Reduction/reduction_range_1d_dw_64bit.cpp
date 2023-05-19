// RUN: %{build} -DENABLE_64_BIT=true -o %t.out %if any-device-is-cuda %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60 %}
// RUN: %{run} %t.out

#include "reduction_range_1d_dw.cpp"
