// REQUIRES: opencl, any-device-is-gpu, any-device-is-cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run} %t.out
