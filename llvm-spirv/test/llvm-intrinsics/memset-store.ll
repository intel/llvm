; Test that memset is lowered to a single OpStore of a composite constant,
; with non-zero fill.
; Also test the global_device & global_host address spaces are translated
; correctly so that the target pointer matches the (bitcast) store pointer,
; with and without SPV_INTEL_usm_storage_classes extension.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV --implicit-check-not=Variable --implicit-check-not=CopyMemorySized --implicit-check-not="Capability USMStorageClassesINTEL"
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-GLOBAL

; TODO: currently the Translator has a bug that it doesn't emit "OpExtension SPV_INTEL_usm_storage_classes"
; when the extension is enabled, therefore spirv-val will reject it. Re-enable spirv-val once this bug is
; fixed in the translator.
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_usm_storage_classes -o %t.usm.spv
; RUN: llvm-spirv %t.usm.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV-USM
; RUN: llvm-spirv -r %t.usm.spv -o - | llvm-dis | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-USM
; RUNx: spirv-val %t.usm.spv

; Without SPV_INTEL_usm_storage_classes both USM address spaces are lowered to
; CrossWorkgroup (storage class 5), so all three kernels share one destination
; pointer type, and fill_device and fill_host share one store pointer type.
; CHECK-SPIRV: TypeInt [[#ByteTy:]] 8 0
; CHECK-SPIRV: Constant [[#]] [[#Length:]] 3
; CHECK-SPIRV: Constant [[#ByteTy]] [[#Byte:]] 42
; CHECK-SPIRV: TypePointer [[#BytePtrTy:]] 5 [[#ByteTy]]
; CHECK-SPIRV: TypeArray [[#ArrayTy:]] [[#ByteTy]] [[#Length]]
; CHECK-SPIRV: TypePointer [[#StorePtrTy:]] 5 [[#ArrayTy]]
; CHECK-SPIRV: TypeArray [[#UsmArrayTy:]] [[#ByteTy]] [[#]]
; CHECK-SPIRV: TypePointer [[#UsmStorePtrTy:]] 5 [[#UsmArrayTy]]
; CHECK-SPIRV: ConstantComposite [[#ArrayTy]] [[#Init:]] [[#Byte]] [[#Byte]] [[#Byte]]
; CHECK-SPIRV: ConstantNull [[#UsmArrayTy]] [[#DeviceInit:]]
; CHECK-SPIRV: ConstantNull [[#UsmArrayTy]] [[#HostInit:]]

; CHECK-SPIRV: FunctionParameter [[#BytePtrTy]] [[#Base:]]
; CHECK-SPIRV: PtrAccessChain [[#BytePtrTy]] [[#OffsetPtr:]] [[#Base]] [[#]]
; CHECK-SPIRV: Bitcast [[#StorePtrTy]] [[#ArrayPtr:]] [[#OffsetPtr]]
; CHECK-SPIRV: Store [[#ArrayPtr]] [[#Init]] 3 1

; CHECK-SPIRV: FunctionParameter [[#BytePtrTy]] [[#Device:]]
; CHECK-SPIRV: Bitcast [[#UsmStorePtrTy]] [[#DeviceStorePtr:]] [[#Device]]
; CHECK-SPIRV: Store [[#DeviceStorePtr]] [[#DeviceInit]] 2 4

; CHECK-SPIRV: FunctionParameter [[#BytePtrTy]] [[#Host:]]
; CHECK-SPIRV: Bitcast [[#UsmStorePtrTy]] [[#HostStorePtr:]] [[#Host]]
; CHECK-SPIRV: Store [[#HostStorePtr]] [[#HostInit]] 2 4

; CHECK-LLVM: %[[Dest:[a-zA-Z0-9.]+]] = getelementptr i8, ptr addrspace(1) %base, i64 1
; CHECK-LLVM: %[[#StorePtr:]] = bitcast ptr addrspace(1) %[[Dest]] to ptr addrspace(1)
; CHECK-LLVM: store volatile [3 x i8] c"***", ptr addrspace(1) %[[#StorePtr]], align 1

; The global_device and global_host destinations were lowered to CrossWorkgroup,
; so they come back as global (addrspace(1)) pointers.
; CHECK-LLVM-GLOBAL: define spir_kernel void @fill_device(ptr addrspace(1) %dst)
; CHECK-LLVM-GLOBAL: %[[#DeviceStorePtr:]] = bitcast ptr addrspace(1) %dst to ptr addrspace(1)
; CHECK-LLVM-GLOBAL: store [8 x i8] zeroinitializer, ptr addrspace(1) %[[#DeviceStorePtr]], align 4
; CHECK-LLVM-GLOBAL: define spir_kernel void @fill_host(ptr addrspace(1) %dst)
; CHECK-LLVM-GLOBAL: %[[#HostStorePtr:]] = bitcast ptr addrspace(1) %dst to ptr addrspace(1)
; CHECK-LLVM-GLOBAL: store [8 x i8] zeroinitializer, ptr addrspace(1) %[[#HostStorePtr]], align 4

; With SPV_INTEL_usm_storage_classes enabled each target pointer keeps its own
; storage class, DeviceOnlyINTEL (5936) and HostOnlyINTEL (5937), and so does
; its store pointer.
; CHECK-SPIRV-USM: TypePointer [[#DeviceBytePtrTy:]] 5936 [[#]]
; CHECK-SPIRV-USM: TypeArray [[#UsmArrayTyE:]] [[#]] [[#]]
; CHECK-SPIRV-USM: TypePointer [[#DeviceStorePtrTy:]] 5936 [[#UsmArrayTyE]]
; CHECK-SPIRV-USM: TypePointer [[#HostBytePtrTy:]] 5937 [[#]]
; CHECK-SPIRV-USM: TypePointer [[#HostStorePtrTy:]] 5937 [[#UsmArrayTyE]]
; CHECK-SPIRV-USM: FunctionParameter [[#DeviceBytePtrTy]] [[#DeviceE:]]
; CHECK-SPIRV-USM: Bitcast [[#DeviceStorePtrTy]] [[#DeviceStorePtrE:]] [[#DeviceE]]
; CHECK-SPIRV-USM: Store [[#DeviceStorePtrE]] [[#]] 2 4
; CHECK-SPIRV-USM: FunctionParameter [[#HostBytePtrTy]] [[#HostE:]]
; CHECK-SPIRV-USM: Bitcast [[#HostStorePtrTy]] [[#HostStorePtrE:]] [[#HostE]]
; CHECK-SPIRV-USM: Store [[#HostStorePtrE]] [[#]] 2 4

; CHECK-LLVM-USM: define spir_kernel void @fill_device(ptr addrspace(5) %dst)
; CHECK-LLVM-USM: %[[#DeviceStorePtrE:]] = bitcast ptr addrspace(5) %dst to ptr addrspace(5)
; CHECK-LLVM-USM: store [8 x i8] zeroinitializer, ptr addrspace(5) %[[#DeviceStorePtrE]], align 4
; CHECK-LLVM-USM: define spir_kernel void @fill_host(ptr addrspace(6) %dst)
; CHECK-LLVM-USM: %[[#HostStorePtrE:]] = bitcast ptr addrspace(6) %dst to ptr addrspace(6)
; CHECK-LLVM-USM: store [8 x i8] zeroinitializer, ptr addrspace(6) %[[#HostStorePtrE]], align 4

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @fill(ptr addrspace(1) %base) {
entry:
  %dst = getelementptr i8, ptr addrspace(1) %base, i64 1
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 1 %dst, i8 42, i64 3, i1 true)
  ret void
}

define spir_kernel void @fill_device(ptr addrspace(5) %dst) {
entry:
  call void @llvm.memset.p5.i64(ptr addrspace(5) align 4 %dst, i8 0, i64 8, i1 false)
  ret void
}

define spir_kernel void @fill_host(ptr addrspace(6) %dst) {
entry:
  call void @llvm.memset.p6.i64(ptr addrspace(6) align 4 %dst, i8 0, i64 8, i1 false)
  ret void
}

declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1 immarg)
declare void @llvm.memset.p5.i64(ptr addrspace(5), i8, i64, i1 immarg)
declare void @llvm.memset.p6.i64(ptr addrspace(6), i8, i64, i1 immarg)
