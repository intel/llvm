// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple spirv64 -emit-llvm %s -fsycl-is-device -o - | FileCheck %s


// CHECK: define spir_func void @_Z9test_castPi
// CHECK: call noundef ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit.p1.p4
// CHECK: call noundef ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit.p3.p4
// CHECK: call noundef ptr @llvm.spv.generic.cast.to.ptr.explicit.p0.p4
// CHECK: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(1)
// CHECK: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(3)
// CHECK: addrspacecast ptr addrspace(4) %{{.*}} to ptr
__attribute__((sycl_device))
void test_cast(int* p) {
  __spirv_GenericCastToPtrExplicit_ToGlobal(p, 5);
  __spirv_GenericCastToPtrExplicit_ToLocal(p, 4);
  __spirv_GenericCastToPtrExplicit_ToPrivate(p, 7);
  __spirv_GenericCastToPtr_ToGlobal(p, 5);
  __spirv_GenericCastToPtr_ToLocal(p, 4);
  __spirv_GenericCastToPtr_ToPrivate(p, 7);
}
