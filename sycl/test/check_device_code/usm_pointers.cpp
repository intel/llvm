// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -emit-llvm %s -S -o %t.ll -I %sycl_include -Wno-sycl-strict -Xclang -verify-ignore-unexpected=note,warning -Xclang -disable-llvm-passes
// RUN: FileCheck %s --input-file %t.ll --check-prefixes=CHECK,CHECK-DISABLE
// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -emit-llvm %s -S -o %t.ll -I %sycl_include -Wno-sycl-strict -Xclang -verify-ignore-unexpected=note,warning -Xclang -disable-llvm-passes -D__ENABLE_USM_ADDR_SPACE__
// RUN: FileCheck %s --input-file %t.ll --check-prefixes=CHECK,CHECK-ENABLE
//
// Check the address space of the pointer in multi_ptr class
//
// CHECK-DISABLE: %[[DEVPTR_T:.*]] = type { ptr addrspace(1) }
// CHECK-DISABLE: %[[HOSTPTR_T:.*]] = type { ptr addrspace(1) }
// CHECK-ENABLE: %[[DEVPTR_T:.*]] = type { ptr addrspace(5) }
// CHECK-ENABLE: %[[HOSTPTR_T:.*]] = type { ptr addrspace(6) }
//
// CHECK-LABEL: define {{.*}} spir_func noundef ptr addrspace(4) @{{.*}}multi_ptr{{.*}}
// CHECK: %[[M_PTR:.*]] = getelementptr inbounds %[[DEVPTR_T]]
// CHECK-DISABLE-NEXT: %[[DEVLOAD:[0-9]+]] = load ptr addrspace(1), ptr addrspace(4) %[[M_PTR]]
// CHECK-DISABLE-NEXT: %[[DEVCAST:[0-9]+]] = addrspacecast ptr addrspace(1) %[[DEVLOAD]] to ptr addrspace(4)
// CHECK-ENABLE-NEXT: %[[DEVLOAD:[0-9]+]] = load ptr addrspace(5), ptr addrspace(4) %[[M_PTR]]
// CHECK-ENABLE-NEXT: %[[DEVCAST:[0-9]+]] = addrspacecast ptr addrspace(5) %[[DEVLOAD]] to ptr addrspace(4)
// ret ptr addrspace(4) %[[DEVCAST]]
//
// CHECK-LABEL: define {{.*}} spir_func noundef ptr addrspace(4) @{{.*}}multi_ptr{{.*}}
// CHECK: %[[M_PTR]] = getelementptr inbounds %[[HOSTPTR_T]]
// CHECK-DISABLE-NEXT: %[[HOSTLOAD:[0-9]+]] = load ptr addrspace(1), ptr addrspace(4) %[[M_PTR]]
// CHECK-DISABLE-NEXT: %[[HOSTCAST:[0-9]+]] = addrspacecast ptr addrspace(1) %[[HOSTLOAD]] to ptr addrspace(4)
// CHECK-ENABLE-NEXT: %[[HOSTLOAD:[0-9]+]] = load ptr addrspace(6), ptr addrspace(4) %[[M_PTR]]
// CHECK-ENABLE-NEXT: %[[HOSTCAST:[0-9]+]] = addrspacecast ptr addrspace(6) %[[HOSTLOAD]] to ptr addrspace(4)
// ret ptr addrspace(4) %[[HOSTCAST]]

#include <sycl/sycl.hpp>

using namespace sycl;

SYCL_EXTERNAL void usm_pointers() {
  void *Ptr = nullptr;
  ext::intel::device_ptr<void> DevPtr(Ptr);
  ext::intel::host_ptr<void> HostPtr(Ptr);
  global_ptr<void> GlobPtr = global_ptr<void>(DevPtr);
  GlobPtr = global_ptr<void>(HostPtr);
}