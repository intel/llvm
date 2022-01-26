// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -emit-llvm %s -S -o %t.ll -I %sycl_include -Wno-sycl-strict -Xclang -verify-ignore-unexpected=note,warning -Xclang -disable-llvm-passes
// RUN: FileCheck %s --input-file %t.ll --check-prefixes=CHECK,CHECK-DISABLE
// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -emit-llvm %s -S -o %t.ll -I %sycl_include -Wno-sycl-strict -Xclang -verify-ignore-unexpected=note,warning -Xclang -disable-llvm-passes -D__ENABLE_USM_ADDR_SPACE__
// RUN: FileCheck %s --input-file %t.ll --check-prefixes=CHECK,CHECK-ENABLE
//
// Check the address space of the pointer in multi_ptr class
//
// CHECK-DISABLE: %[[DEVPTR_T:.*]] = type { i8 addrspace(1)* }
// CHECK-DISABLE: %[[HOSTPTR_T:.*]] = type { i8 addrspace(1)* }
// CHECK-ENABLE: %[[DEVPTR_T:.*]] = type { i8 addrspace(5)* }
// CHECK-ENABLE: %[[HOSTPTR_T:.*]] = type { i8 addrspace(6)* }
//
// CHECK-LABEL: define {{.*}} spir_func noundef i8 addrspace(4)* @{{.*}}multi_ptr{{.*}}
// CHECK: %[[M_PTR:.*]] = getelementptr inbounds %[[DEVPTR_T]]
// CHECK-DISABLE-NEXT: %[[DEVLOAD:[0-9]+]] = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %[[M_PTR]]
// CHECK-DISABLE-NEXT: %[[DEVCAST:[0-9]+]] = addrspacecast i8 addrspace(1)* %[[DEVLOAD]] to i8 addrspace(4)*
// CHECK-ENABLE-NEXT: %[[DEVLOAD:[0-9]+]] = load i8 addrspace(5)*, i8 addrspace(5)* addrspace(4)* %[[M_PTR]]
// CHECK-ENABLE-NEXT: %[[DEVCAST:[0-9]+]] = addrspacecast i8 addrspace(5)* %[[DEVLOAD]] to i8 addrspace(4)*
// ret i8 addrspace(4)* %[[DEVCAST]]
//
// CHECK-LABEL: define {{.*}} spir_func noundef i8 addrspace(4)* @{{.*}}multi_ptr{{.*}}
// CHECK: %[[M_PTR]] = getelementptr inbounds %[[HOSTPTR_T]]
// CHECK-DISABLE-NEXT: %[[HOSTLOAD:[0-9]+]] = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(4)* %[[M_PTR]]
// CHECK-DISABLE-NEXT: %[[HOSTCAST:[0-9]+]] = addrspacecast i8 addrspace(1)* %[[HOSTLOAD]] to i8 addrspace(4)*
// CHECK-ENABLE-NEXT: %[[HOSTLOAD:[0-9]+]] = load i8 addrspace(6)*, i8 addrspace(6)* addrspace(4)* %[[M_PTR]]
// CHECK-ENABLE-NEXT: %[[HOSTCAST:[0-9]+]] = addrspacecast i8 addrspace(6)* %[[HOSTLOAD]] to i8 addrspace(4)*
// ret i8 addrspace(4)* %[[HOSTCAST]]

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  cl::sycl::queue queue;
  {
    queue.submit([&](cl::sycl::handler &cgh) {
      cgh.single_task<class check_adress_space>([=]() {
        void *Ptr = nullptr;
        device_ptr<void> DevPtr(Ptr);
        host_ptr<void> HostPtr(Ptr);
        global_ptr<void> GlobPtr = global_ptr<void>(DevPtr);
        GlobPtr = global_ptr<void>(HostPtr);
      });
    });
    queue.wait();
  }

  return 0;
}
