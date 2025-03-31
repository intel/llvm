// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17515
// There are no CUDA libs on win machines, so the compilation fails.

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: %if !cuda %{ not %} %{run} %t.out
