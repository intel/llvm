// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/is_compatible_with_env.cpp -o %t.out
// RUN: %{run} not %t.out

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %S/Inputs/is_compatible_with_env.cpp -o %t.out
// RUN: %{run} %t.out
