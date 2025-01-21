// REQUIRES: cuda, opencl, gpu, cpu
// REQUIRES: build-and-run-mode

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=cuda:gpu %{run} not %t.out
