// REQUIRES: cpu, gpu, accelerator

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga %S/Inputs/is_compatible_with_env.cpp -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:fpga %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run} %t.out
