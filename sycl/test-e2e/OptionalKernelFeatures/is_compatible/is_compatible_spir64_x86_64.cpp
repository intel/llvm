// REQUIRES: opencl-aot, any-device-is-cpu, any-device-is-gpu, any-device-is-level_zero

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run-unfiltered-devices} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu %{run-unfiltered-devices} not %t.out
