// REQUIRES: opencl-aot, ocloc, opencl-cpu-rt

// RUN: %{run-aux} %clangxx -fsycl -fsycl-targets=spir64_x86_64,%{gpu_aot_target} %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !(level_zero || opencl) %{ not %} %{run} %t.out
