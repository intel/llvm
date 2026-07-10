// REQUIRES: ocloc
// REQUIRES: intel-gpu-aot-targets || !new-offload-model

// RUN: %clangxx -fsycl %{gpu_aot_opts} %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !(level_zero || opencl && gpu) %{ not %} %{run} %t.out
