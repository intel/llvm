// REQUIRES: hip_amd, opencl, gpu, cpu

// RUN: %clangxx -fsycl -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx906 -fsycl-targets=amdgcn-amd-amdhsa %S/Inputs/is_compatible_with_env.cpp -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=hip:gpu %{run} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %{run} not %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run} not %t.out
