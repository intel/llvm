
// On CUDA, the test behaves differently depending on whether it is compiled for
// sm_xx>=sm_80 or not:
// + sm_80 and above uses some native bfloat16 math instructions
// + below sm_80 always uses generic impls

// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}
// RUN: %clangxx -fsycl %{sycl_target_opts} %if target-nvidia %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 %} %s -o %t.out %{mathflags}
// RUN: %{run} %t.out

// Flaky timeout on CPU. Enable when fixed.
// Depends on SPIR-V Backend & run-time drivers version.
// UNSUPPORTED: spirv-backend && cpu
// UNSUPPORTED-TRACKER: CMPLRLLVM-64705

#include "bfloat16_builtins.hpp"

int main() {

  test();
  return 0;
}
