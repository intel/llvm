// REQUIRES: opencl-aot, ocloc

// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17515
// There is no CPU device on win yet, so opencl-aot fails to compile the kernel.

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !(level_zero || opencl) %{ not %} %{run} %t.out
