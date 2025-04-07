// REQUIRES: cuda_dev_kit

// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17515
// CUDA libs are not installed correctly.

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !cuda %{ not %} %{run} %t.out
