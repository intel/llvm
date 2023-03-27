// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Daccessor_new_api_test %S/Inputs/host_task_accessor.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
