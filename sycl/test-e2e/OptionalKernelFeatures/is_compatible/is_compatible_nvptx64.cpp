// REQUIRES: cuda, opencl, gpu, cpu

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=cuda:gpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run} not %t.out
