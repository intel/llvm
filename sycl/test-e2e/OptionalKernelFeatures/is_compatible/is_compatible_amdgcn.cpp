// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17515
// There are no ROCm libs on win machines, so the compilation fails.

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1030 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !hip %{ not %} %{run} %t.out
