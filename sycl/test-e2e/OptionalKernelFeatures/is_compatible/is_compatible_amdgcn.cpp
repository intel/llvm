// REQUIRES: hip_dev_kit

// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17515
// ROCm libraries are not installed correctly.

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1030 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !hip %{ not %} %{run} %t.out
