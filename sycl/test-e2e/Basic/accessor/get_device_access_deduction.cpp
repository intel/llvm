// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -Dbuffer_new_api_test %S/Inputs/device_accessor.cpp -o %t.out
// RUN: %{run} %t.out
