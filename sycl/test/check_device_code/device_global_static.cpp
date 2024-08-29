// Tests that the llvm.compiler.used symbol, which is used to implement static
// device globals, is removed at some point in compilation. For SPIR-V this
// symbol is removed at sycl-post-link and for NVPTX/AMDGCN it is removed at
// lowering.

// UNSUPPORTED: windows

// RUN: %clangxx -fsycl %s -o %t
// RUN: strings %t | not grep "llvm.compiler.used"

// RUN: %if cuda %{ %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s -o %t %}
// RUN: %if cuda %{ strings %t | not grep "llvm.compiler.used" %}

// RUN: %if hip_amd %{ %clangxx -fsycl -fsycl-targets=amd_gpu_gfx906 %s -o %t %}
// RUN: %if hip_amd %{ strings %t | not grep "llvm.compiler.used" %}

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

static device_global<int> DeviceGlobalVar;

int main() {
  sycl::queue{}.single_task([=] { volatile int ReadVal = DeviceGlobalVar; });
}
