// REQUIRES: ocloc, gpu, level_zero, opencl

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend "-device *" %S/Inputs/is_compatible_with_env.cpp -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zerp:gpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run} %t.out
