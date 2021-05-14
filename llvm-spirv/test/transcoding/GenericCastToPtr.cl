// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -emit-llvm-bc -fdeclare-opencl-builtins -finclude-default-header %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-OCL
// RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-SPV
// RUN: llvm-spirv %t.rev.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

// CHECK-SPIRV: 4 GenericCastToPtr

// CHECK-LLVM-LABEL: @testGenericCastToPtrGlobal
// CHECK-LLVM: %0 = addrspacecast <2 x i16> addrspace(4)* %a to <2 x i16> addrspace(1)*

global short2 *testGenericCastToPtrGlobal(generic short2 *a) {
  return (global short2 *)a;
}

// CHECK-SPIRV: 4 GenericCastToPtr

// CHECK-LLVM-LABEL: @testGenericCastToPtrLocal
// CHECK-LLVM: %0 = addrspacecast <2 x i16> addrspace(4)* %a to <2 x i16> addrspace(3)*

local short2 *testGenericCastToPtrLocal(generic short2 *a) {
  return (local short2 *)a;
}

// CHECK-SPIRV: 4 GenericCastToPtr

// CHECK-LLVM-LABEL: @testGenericCastToPtrPrivate
// CHECK-LLVM: %0 = addrspacecast <2 x i16> addrspace(4)* %a to <2 x i16>*

private short2 *testGenericCastToPtrPrivate(generic short2 *a) {
  return (private short2 *)a;
}

// CHECK-SPIRV: 5 GenericCastToPtrExplicit

// CHECK-LLVM-LABEL: @testGenericCastToPtrExplicitGlobal
// CHECK-LLVM: %[[VoidPtrCast:[0-9]+]] = bitcast <2 x i16> addrspace(4)* %a to i8 addrspace(4)*
// CHECK-LLVM-NEXT: %[[AddrSpaceCast:[0-9]+]] = bitcast i8 addrspace(4)* %[[VoidPtrCast]] to i8 addrspace(4)*
// CHECK-LLVM-OCL-NEXT: %{{[0-9a-zA-Z.]+}} = call spir_func i8 addrspace(1)* @__to_global(i8 addrspace(4)* %[[AddrSpaceCast]])
// CHECK-LLVM-SPV-NEXT: %{{[0-9a-zA-Z.]+}} = call spir_func i8 addrspace(1)* @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPU3AS4ci(i8 addrspace(4)* %[[AddrSpaceCast]], i32
// CHECK-LLVM: bitcast i8 addrspace(1)* %{{[0-9]+}} to <2 x i16> addrspace(1)*

global short2 *testGenericCastToPtrExplicitGlobal(generic short2 *a) {
  return to_global(a);
}

// CHECK-SPIRV: 5 GenericCastToPtrExplicit

// CHECK-LLVM-LABEL: @testGenericCastToPtrExplicitLocal
// CHECK-LLVM: %[[VoidPtrCast:[0-9]+]] = bitcast <2 x i16> addrspace(4)* %a to i8 addrspace(4)*
// CHECK-LLVM-NEXT: %[[AddrSpaceCast:[0-9]+]] = bitcast i8 addrspace(4)* %[[VoidPtrCast]] to i8 addrspace(4)*
// CHECK-LLVM-OCL-NEXT: %{{[0-9a-zA-Z.]+}} = call spir_func i8 addrspace(3)* @__to_local(i8 addrspace(4)* %[[AddrSpaceCast]])
// CHECK-LLVM-SPV-NEXT: %{{[0-9a-zA-Z.]+}} = call spir_func i8 addrspace(3)* @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPU3AS4ci(i8 addrspace(4)* %[[AddrSpaceCast]], i32
// CHECK-LLVM: bitcast i8 addrspace(3)* %{{[0-9]+}} to <2 x i16> addrspace(3)*

local short2 *testGenericCastToPtrExplicitLocal(generic short2 *a) {
  return to_local(a);
}

// CHECK-SPIRV: 5 GenericCastToPtrExplicit

// CHECK-LLVM-LABEL: @testGenericCastToPtrExplicitPrivate
// CHECK-LLVM: %[[VoidPtrCast:[0-9]+]] = bitcast <2 x i16> addrspace(4)* %a to i8 addrspace(4)*
// CHECK-LLVM-NEXT: %[[AddrSpaceCast:[0-9]+]] = bitcast i8 addrspace(4)* %[[VoidPtrCast]] to i8 addrspace(4)*
// CHECK-LLVM-OCL-NEXT: %{{[0-9a-zA-Z.]+}} = call spir_func i8* @__to_private(i8 addrspace(4)* %[[AddrSpaceCast]])
// CHECK-LLVM-SPV-NEXT: %{{[0-9a-zA-Z.]+}} = call spir_func i8* @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePU3AS4ci(i8 addrspace(4)* %[[AddrSpaceCast]], i32
// CHECK-LLVM: bitcast i8* %{{[0-9]+}} to <2 x i16>*

private short2 *testGenericCastToPtrExplicitPrivate(generic short2 *a) {
  return to_private(a);
}
