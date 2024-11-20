// REQUIRES: hip_amd, any-device-is-opencl, any-device-is-gpu, any-device-is-cpu

// RUN: %clangxx -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx906 -fsycl-targets=amdgcn-amd-amdhsa %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=hip:gpu %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run-unfiltered-devices} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out
