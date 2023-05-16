
// does not actually require OpenCL or GPU. Just testing parsing.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %S/Inputs/trivial.cpp -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="OPENCL:*" %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="opencl:*" %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:GPU" %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %t.out
