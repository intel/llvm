// Tests that the llvm.compiler.used symbol, which is used to implement static
// device globals, is removed at some point in compilation. For SPIR-V this
// symbol is removed at sycl-post-link and for NVPTX/AMDGCN it is removed at
// lowering.
//
// It also checks that the symbol can be found in an object file for a given
// triple, thus validating that `llvm-strings` can successfully be used to
// check for the presence of the symbol.

// UNSUPPORTED: windows

// RUN: %clangxx -fsycl -fsycl-device-only %s -o %t
// RUN: llvm-strings %t | grep "llvm.compiler.used"
// RUN: %clangxx -fsycl %s -o %t
// RUN: llvm-strings %t | not grep "llvm.compiler.used"

// RUN: %if cuda %{ %clangxx -fsycl -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda %s -o %t %}
// RUN: %if cuda %{ llvm-strings %t | grep "llvm.compiler.used" %}
// RUN: %if cuda %{ %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %s -o %t %}
// RUN: %if cuda %{ llvm-strings %t | not grep "llvm.compiler.used" %}

// RUN: %if hip_amd %{ %clangxx -fsycl -fsycl-device-only -fsycl-targets=amd_gpu_gfx906 %s -o %t %}
// RUN: %if hip_amd %{ llvm-strings %t | grep "llvm.compiler.used" %}
// RUN: %if hip_amd %{ %clangxx -fsycl -fsycl-targets=amd_gpu_gfx906 %s -o %t %}
// RUN: %if hip_amd %{ llvm-strings %t | not grep "llvm.compiler.used" %}

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

static device_global<int> DeviceGlobalVar;

int main() {
  sycl::queue{}.single_task([=] { volatile int ReadVal = DeviceGlobalVar; });
}
