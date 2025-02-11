// REQUIRES: ocloc, any-device-is-cpu, any-device-is-level_zero

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out
