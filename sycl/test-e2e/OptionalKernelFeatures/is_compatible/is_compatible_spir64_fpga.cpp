// REQUIRES: opencl-aot, any-device-is-accelerator, any-device-is-gpu, any-device-is-cpu

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:fpga %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=*:gpu %{run-unfiltered-devices} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out
