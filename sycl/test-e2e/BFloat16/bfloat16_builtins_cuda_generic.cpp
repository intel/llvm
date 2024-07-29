
// On CUDA, the test behaves differently depending on whether it is compiled for
// sm_xx>=sm_80 or not:
// + sm_80 and above uses some native bfloat16 math instructions
// + below sm_80 always uses generic impls

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// If CUDA, test "new" again for sm_75/generic
// RUN:  %if any-device-is-cuda %{ %if preview-breaking-changes-supported %{  %clangxx -fsycl -fpreview-breaking-changes -fsycl-targets=%{sycl_triple}  -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_75  %s -o %t3.out %{mathflags} %} %}
// RUN:  %if any-device-is-cuda %{ %if preview-breaking-changes-supported %{  %{run} %t3.out  %} %}

// Currently the feature isn't supported on FPGA.
// UNSUPPORTED: accelerator
#include "bfloat16_builtins.hpp"

int main() {

  test();
  return 0;
}
