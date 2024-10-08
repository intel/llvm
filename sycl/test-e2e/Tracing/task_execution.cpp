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
    Q.memset(AllocSrc, 0, 1).wait();
    Q.copy(AllocDst, AllocSrc, 1).wait();
    // CHECK: [SYCL] Queue create:
    // CHECK-DAG:        queue_handle : {{.*}}
    // CHECK-DAG:        queue_id : 1
    // CHECK-DAG:        is_inorder : false
    // CHECK-DAG:        sycl_device : {{.*}}
    // CHECK-DAG:        sycl_device_name : {{.*}}
    // CHECK-DAG:        sycl_context : {{.*}}
    // CHECK-NEXT: [SYCL] Task begin (event={{.*}},instanceID={{.*}})
    // CHECK-DAG:          queue_id : 1
    // CHECK-DAG:          memory_size : 1
    // CHECK-DAG:          value_set : 0
    // CHECK-DAG:          memory_ptr : {{.*}}
    // CHECK-DAG:          sycl_device : {{.*}}
    // CHECK-NEXT: [SYCL] Task end   (event={{.*}},instanceID={{.*}})
    // CHECK-NEXT: [SYCL] Task begin (event={{.*}},instanceID={{.*}})
    // CHECK-DAG:          queue_id : 1
    // CHECK-DAG:          memory_size : 1
    // CHECK-DAG:          dest_memory_ptr : {{.*}}
    // CHECK-DAG:          src_memory_ptr : {{.*}}
    // CHECK-DAG:          sycl_device : {{.*}}
    // CHECK-NEXT: [SYCL] Task end   (event={{.*}},instanceID={{.*}})
    Q.single_task<class E2ETestKernel>([]() {}).wait();
    // CHECK-NEXT: [SYCL] Task begin (event={{.*}},instanceID={{.*}})
    // CHECK-DAG:          enqueue_kernel_data : {{.*}}
    // CHECK-DAG:          sym_column_no : {{.*}}
    // CHECK-DAG:          sym_line_no : 37
    // CHECK-DAG:          sym_source_file_name : {{.*}}task_execution.cpp
    // CHECK-DAG:          queue_id : 1
    // CHECK-DAG:          sym_function_name : typeinfo name for main::E2ETestKernel
    // CHECK-DAG:          from_source : {{.*}}
    // CHECK-DAG:          sycl_device_name : {{.*}}
    // CHECK-DAG:          sycl_device_type : {{.*}}
    // CHECK-DAG:          kernel_name : typeinfo name for main::E2ETestKernel
    // CHECK-DAG:          sycl_device : {{.*}}
    // CHECK-NEXT: [SYCL] Task end   (event={{.*}},instanceID={{.*}})
    // CHECK-NEXT: [SYCL] Queue destroy:
    // CHECK-DAG:        queue_id : 1
    sycl::free(AllocSrc, Q);
    sycl::free(AllocDst, Q);
  }
  return 0;
}
