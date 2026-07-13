// REQUIRES: ocloc

// RUN: %clangxx -fsycl -fsycl-targets=%{intel_gpu_aot_targets} %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !(level_zero || opencl && gpu) %{ not %} %{run} %t.out
