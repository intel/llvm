; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-WO-EXT

; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt --spirv-ext=+SPV_INTEL_non_constant_addrspace_printf
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -to-binary %t.spt -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-WO-EXT: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-WO-EXT: The SPV_INTEL_non_constant_addrspace_printf extension should be allowed to translate this module, because this LLVM module contains the printf function with format string, whose address space is not equal to 2 (constant).

; CHECK-SPIRV: Capability NonConstantAddrspacePrintfINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_non_constant_addrspace_printf"
; CHECK-SPIRV: ExtInstImport [[#ExtInstSetId:]] "OpenCL.std"
; CHECK-SPIRV: TypeInt [[#TypeInt8Id:]] 8 0
; CHECK-SPIRV: TypeInt [[#TypeInt32Id:]] 32 0
; CHECK-SPIRV: TypePointer [[#FunctionStorClassPtrTy:]] 7 [[#TypeInt8Id]]
; CHECK-SPIRV: TypePointer [[#WGStorClassPtrTy:]] 5 [[#TypeInt8Id]]
; CHECK-SPIRV: TypePointer [[#CrossWFStorClassPtrTy:]] 4 [[#TypeInt8Id]]
; CHECK-SPIRV: TypePointer [[#GenericStorCalssPtrTy:]] 8 [[#TypeInt8Id]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#FunctionStorClassPtrTy]] [[#GEP1:]]
; CHECK-SPIRV: ExtInst [[#TypeInt32Id]] [[#]] [[#ExtInstSetId:]] printf [[#GEP1]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#WGStorClassPtrTy]] [[#GEP2:]]
; CHECK-SPIRV: ExtInst [[#TypeInt32Id]] [[#]] [[#ExtInstSetId:]] printf [[#GEP2]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#CrossWFStorClassPtrTy:]] [[#GEP3:]]
; CHECK-SPIRV: ExtInst [[#TypeInt32Id]] [[#]] [[#ExtInstSetId:]] printf [[#GEP3]]
; CHECK-SPIRV: InBoundsPtrAccessChain [[#GenericStorCalssPtrTy:]] [[#GEP4:]]
; CHECK-SPIRV: ExtInst [[#TypeInt32Id]] [[#]] [[#ExtInstSetId:]] printf [[#GEP4]]

; CHECK-LLVM: call spir_func i32 @_Z18__spirv_ocl_printfPc(i8* {{.*}}
; CHECK-LLVM: call spir_func i32 @_Z18__spirv_ocl_printfPU3AS1c(i8 addrspace(1)* {{.*}}
; CHECK-LLVM: call spir_func i32 @_Z18__spirv_ocl_printfPU3AS3c(i8 addrspace(3)* {{.*}}
; CHECK-LLVM: call spir_func i32 @_Z18__spirv_ocl_printfPU3AS4c(i8 addrspace(4)* {{.*}}

; ModuleID = 'non-constant-printf'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

@0 = internal unnamed_addr addrspace(0) constant [6 x i8] c"Test\0A\00", align 1
@1 = internal unnamed_addr addrspace(1) constant [6 x i8] c"Test\0A\00", align 1
@2 = internal unnamed_addr addrspace(3) constant [6 x i8] c"Test\0A\00", align 1
@3 = internal unnamed_addr addrspace(4) constant [6 x i8] c"Test\0A\00", align 1

; Function Attrs: nounwind
define spir_kernel void @test() #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_type_qual !3 !kernel_arg_base_type !3 {
  %1 = getelementptr inbounds [6 x i8], [6 x i8]* @0, i32 0, i32 0
  %2 = call spir_func i32 @_Z18__spirv_ocl_printfPc(i8* %1) #0
  %3 = getelementptr inbounds [6 x i8], [6 x i8] addrspace(1)* @1, i32 0, i32 0
  %4 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS1c(i8 addrspace(1)* %3) #0
  %5 = getelementptr inbounds [6 x i8], [6 x i8] addrspace(3)* @2, i32 0, i32 0
  %6 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS3c(i8 addrspace(3)* %5) #0
  %7 = getelementptr inbounds [6 x i8], [6 x i8] addrspace(4)* @3, i32 0, i32 0
  %8 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS4c(i8 addrspace(4)* %7) #0
  ret void
}

; Function Attrs: nounwind
declare spir_func i32 @_Z18__spirv_ocl_printfPc(i8*) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS1c(i8 addrspace(1)*) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS3c(i8 addrspace(3)*) #0

; Function Attrs: nounwind
declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS4c(i8 addrspace(4)*) #0

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}

!0 = !{i32 1, i32 2}
!1 = !{i32 3, i32 200000}
!2 = !{i32 2, i32 0}
!3 = !{}
!4 = !{i16 7, i16 0}
