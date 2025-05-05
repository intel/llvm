// REQUIRES: cuda_dev_kit

// RUN: %clangxx %cuda_options -fsycl -fsycl-targets=nvptx64-nvidia-cuda %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !cuda %{ not %} %{run} %t.out
