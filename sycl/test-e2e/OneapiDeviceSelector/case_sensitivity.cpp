
// does not actually require OpenCL or GPU. Just testing parsing.

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/trivial.cpp -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="OPENCL:*" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="opencl:*" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:GPU" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out
