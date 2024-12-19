// REQUIRES: ocloc, any-device-is-level_zero, any-device-is-gpu, any-device-is-cpu
// REQUIRES: build-and-run-mode

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu %{run-unfiltered-devices} %t.out