// REQUIRES: cpu
// UNSUPPORTED: windows
// RUN: %{build} -o %t.out
// RUN: %{run} sycl-trace --sycl --print-format=verbose %t.out | FileCheck %s

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

int main() {
  {
    sycl::queue Q;
    unsigned char *AllocSrc = (unsigned char *)sycl::malloc_device(1, Q);
    unsigned char *AllocDst = (unsigned char *)sycl::malloc_device(1, Q);
    Q.submit([&](sycl::handler &cgh)

             { cgh.memset(AllocSrc, 0, 1); })
        .wait();
    // CHECK: [SYCL] Task begin (event={{.*}},instanceID={{.*}})
    // CHECK-DAG:          queue_id : 1
    // CHECK-DAG:          sym_column_no : {{.*}}
    // CHECK-DAG:          sym_function_name : {{.*}}
    // CHECK-DAG:          kernel_name : {{.*}}
    // CHECK-DAG:          sym_source_file_name : {{.*}}task_execution_handler.cpp
    // CHECK-DAG:          sycl_device_name : {{.*}}
    // CHECK-DAG:          sycl_device_type : {{.*}}
    // CHECK-DAG:          sym_line_no : {{.*}}
    // CHECK-DAG:          sycl_device : {{.*}}
    // CHECK-NEXT: [SYCL] Task end   (event={{.*}},instanceID={{.*}})
    // CHECK-NEXT: [SYCL] Task begin (event={{.*}},instanceID={{.*}})
    // CHECK-DAG:          queue_id : 1
    // CHECK-DAG:          sym_column_no : {{.*}}
    // CHECK-DAG:          sym_function_name : {{.*}}
    // CHECK-DAG:          kernel_name : {{.*}}
    // CHECK-DAG:          sym_source_file_name : {{.*}}task_execution_handler.cpp
    // CHECK-DAG:          sycl_device_name : {{.*}}
    // CHECK-DAG:          sycl_device_type : {{.*}}
    // CHECK-DAG:          sym_line_no : {{.*}}
    // CHECK-DAG:          sycl_device : {{.*}}
    // CHECK-NEXT: [SYCL] Task end   (event={{.*}},instanceID={{.*}})
    Q.submit([&](sycl::handler &cgh)

             { cgh.memcpy(AllocDst, AllocSrc, 1); })
        .wait();
    sycl::free(AllocSrc, Q);
    sycl::free(AllocDst, Q);
  }
  return 0;
}
