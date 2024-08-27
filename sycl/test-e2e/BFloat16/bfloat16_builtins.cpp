
// On CUDA, the test behaves differently depending on whether it is compiled for
// sm_xx>=sm_80 or not:
// + sm_80 and above uses some native bfloat16 math instructions
// + below sm_80 always uses generic impls

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %if any-device-is-cuda %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 %} %s -o %t.out %{mathflags}
// RUN: %{run} %t.out

// Test "new" (ABI breaking) for all platforms ( sm_80/native if CUDA )
// RUN:  %if preview-breaking-changes-supported %{  %clangxx -fsycl -fpreview-breaking-changes -fsycl-targets=%{sycl_triple} %if any-device-is-cuda %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 %} %s -o %t2.out %{mathflags} %}
// RUN:  %if preview-breaking-changes-supported %{  %{run} %t2.out  %}

// Currently the feature isn't supported on FPGA.
// UNSUPPORTED: accelerator
#include "bfloat16_builtins.hpp"

int main() {

  test();
  return 0;
}
