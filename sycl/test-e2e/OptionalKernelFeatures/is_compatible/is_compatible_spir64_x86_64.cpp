// REQUIRES: opencl-aot, cpu, gpu, level_zero
// REQUIRES: build-and-run-mode

// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu %{run} not %t.out
