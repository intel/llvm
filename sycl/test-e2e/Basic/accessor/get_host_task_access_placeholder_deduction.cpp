// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} -Dbuffer_placeholder_new_api_test %S/Inputs/host_task_accessor.cpp -o %t.out
// RUN: %{run} %t.out
