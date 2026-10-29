// REQUIRES: ocloc, target-spir

// RUN: %clangxx -fsycl -fsycl-targets=%{gpu_aot_target} %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !(level_zero || opencl && gpu) %{ not %} %{run} %t.out
