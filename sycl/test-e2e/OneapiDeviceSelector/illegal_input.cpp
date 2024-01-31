
// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %S/Inputs/trivial.cpp -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="macaroni:*" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=":" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:." %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="macaroni_level_zero:." %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:macaroni_gpu" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:0..0" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:::gpu" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:.1" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:acc" %{run-unfiltered-devices} %t.out
// XFAIL: *

// Calling ONEAPI_DEVICE_SELECTOR with an illegal input should result in an
// error.
