// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST SYCL_PRINT_EXECUTION_GRAPH=after_addCG %t.out
// RUN: cat graph_2after_addCG.dot | FileCheck %s
// RUN: rm *.dot

// This test is executed on host device only because flush buffer initialized
// via a separate call group task for this device only.

#include <CL/sycl.hpp>

using namespace cl;

int main() {
  {
    sycl::queue Queue;
    Queue.submit([&](sycl::handler &cgh) {
      sycl::stream Out(100, 100, cgh);
      cgh.single_task<class test_flush_buf_init_dep>(
          [=]() { Out << "Hello world!" << sycl::endl; });
    });
    Queue.wait();
  }
  // CHECK: [[MAIN_CG:"0x[0-9a-f]+"]] {{.*}}EXEC CG ON HOST{{.*}}test_flush_buf_init_dep
  // First 3 lines show dependencies on global buffer related commands
  // CHECK:   [[MAIN_CG]]
  // CHECK:   [[MAIN_CG]]
  // CHECK:   [[MAIN_CG]]
  // CHECK:   [[MAIN_CG]] -> [[EMPTY_NODE:"0x[0-9a-f]+"]] [ label = "Access mode: read_write\nMemObj: [[FLUSHBUF_MEMOBJ:0x[0-9a-f]+]]
  // CHECK: [[EMPTY_NODE]] {{.*}}EMPTY NODE
  // CHECK:   [[EMPTY_NODE]] -> [[FILL_TASK:"0x[0-9a-f]+"]] [ label = "Access mode: discard_write\nMemObj: [[FLUSHBUF_MEMOBJ]]
  // CHECK: [[FILL_TASK]] {{.*}}EXEC CG ON HOST\nCG type: host task
  // CHECK:   [[FILL_TASK]] -> [[ALLOC_TASK:"0x[0-9a-f]+"]] [ label = "Access mode: discard_write\nMemObj: [[FLUSHBUF_MEMOBJ]]
  // CHECK: [[ALLOC_TASK]] {{.*}}ALLOCA ON HOST\n MemObj : [[FLUSHBUF_MEMOBJ]]
}
  return 0;
}
