// REQUIRES: any-device-is-cuda, any-device-is-opencl, any-device-is-gpu, any-device-is-cpu

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=cuda:gpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run-unfiltered-devices} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out
