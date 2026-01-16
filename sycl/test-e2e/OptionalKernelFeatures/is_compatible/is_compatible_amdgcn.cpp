// REQUIRES: hip_dev_kit

// XFAIL: new-offload-model
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20797

// RUN: %clangxx %hip_options -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1030 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !hip %{ not %} %{run} %t.out
