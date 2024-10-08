; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR


; CHECK-SPV-IR: %"[[CL_PIPE_STORAGE_NAME:[^"]+]]" = type { ptr addrspace(1) }
; CHECK-SPV-IR: %"[[CL_READ_PIPE_NAME:[^"]+read>]]" = type { target("spirv.Pipe", 0) }
; CHECK-SPV-IR: %"[[CL_WRITE_PIPE_NAME:[^"]+write>]]" = type { target("spirv.Pipe", 1) }


; CHECK-SPIRV: Capability Pipes
; CHECK-SPIRV: Capability PipeStorage

; CHECK-SPIRV: Name [[PIPE_STORAGE_ID:[0-9]+]] "mygpipe"
; CHECK-SPIRV: Name [[READ_PIPE_WRAPPER_ID:[0-9]+]] "myrpipe"
; CHECK-SPIRV: Name [[WRITE_PIPE_WRAPPER_ID:[0-9]+]] "mywpipe"
; CHECK-SPIRV: Name [[WRITE_PIPE_WRAPPER_CTOR:[0-9]+]] "_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE1EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE1EEE"
; CHECK-SPIRV: Name [[READ_PIPE_WRAPPER_CTOR:[0-9]+]] "_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE0EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE0EEE"

; CHECK-SPIRV: TypeInt [[INT_T:[0-9]+]] 32 0
; CHECK-SPIRV: Constant [[INT_T]] [[CONSTANT_ZERO_ID:[0-9]+]] 0

; CHECK-SPIRV: TypePipe [[READ_PIPE:[0-9]+]] 0
; CHECK-SPIRV: TypeStruct [[READ_PIPE_WRAPPER:[0-9]+]] [[READ_PIPE]]
; CHECK-SPIRV: TypePointer [[READ_PIPE_WRAPPER_PTR:[0-9]+]] 7 [[READ_PIPE_WRAPPER]]
; CHECK-SPIRV: TypePipe [[WRITE_PIPE:[0-9]+]] 1
; CHECK-SPIRV: TypeStruct [[WRITE_PIPE_WRAPPER:[0-9]+]] [[WRITE_PIPE]]

; CHECK-SPIRV: TypePointer [[WRITE_PIPE_WRAPPER_PTR:[0-9]+]] 7 [[WRITE_PIPE_WRAPPER]]


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%spirv.ConstantPipeStorage = type { i32, i32, i32 }
%"class.cl::pipe_storage<int __attribute__((ext_vector_type(4))), 1>" = type { ptr addrspace(1) }
%"class.cl::pipe<int __attribute__((ext_vector_type(4))), cl::pipe_access::read>" = type { target("spirv.Pipe", 0) }
%"class.cl::pipe<int __attribute__((ext_vector_type(4))), cl::pipe_access::write>" = type { target("spirv.Pipe", 1) }

@_ZN2cl9__details29OpConstantPipeStorage_CreatorILi16ELi16ELi1EE5valueE = linkonce_odr addrspace(1) global %spirv.ConstantPipeStorage { i32 16, i32 16, i32 1 }, align 4
@mygpipe = addrspace(1) global %"class.cl::pipe_storage<int __attribute__((ext_vector_type(4))), 1>" { ptr addrspace(1) @_ZN2cl9__details29OpConstantPipeStorage_CreatorILi16ELi16ELi1EE5valueE }, align 4


; Function Attrs: nounwind
define spir_kernel void @worker() {
entry:
  ; CHECK-SPV-IR: %myrpipe = alloca %"[[CL_READ_PIPE_NAME]]", align 4
  ; CHECK-SPV-IR: %mywpipe = alloca %"[[CL_WRITE_PIPE_NAME]]", align 4

  ; CHECK-SPIRV: Variable [[READ_PIPE_WRAPPER_PTR]] [[READ_PIPE_WRAPPER_ID]] 7
  ; CHECK-SPIRV: Variable [[WRITE_PIPE_WRAPPER_PTR]] [[WRITE_PIPE_WRAPPER_ID]] 7

  %myrpipe = alloca %"class.cl::pipe<int __attribute__((ext_vector_type(4))), cl::pipe_access::read>", align 4
  %mywpipe = alloca %"class.cl::pipe<int __attribute__((ext_vector_type(4))), cl::pipe_access::write>", align 4


  ; CHECK-SPV-IR: %[[ID0:[0-9]+]] = addrspacecast ptr addrspace(1) @mygpipe to ptr addrspace(4)
  ; CHECK-SPV-IR: %[[ID1:[0-9]+]] = getelementptr %"[[CL_PIPE_STORAGE_NAME]]", ptr addrspace(4) %[[ID0]], i32 0, i32 0

  ; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[SPIRV0:[0-9]+]] [[PIPE_STORAGE_ID]]
  ; CHECK-SPIRV: PtrAccessChain {{[0-9]+}} [[SPIRV1:[0-9]+]] [[SPIRV0]] [[CONSTANT_ZERO_ID]] [[CONSTANT_ZERO_ID]]

  %0 = addrspacecast ptr addrspace(1) @mygpipe to ptr addrspace(4)
  %1 = getelementptr %"class.cl::pipe_storage<int __attribute__((ext_vector_type(4))), 1>", ptr addrspace(4) %0, i32 0, i32 0


  ; CHECK-SPV-IR: %[[PIPE_STORAGE_1:[0-9]+]] = load ptr addrspace(1), ptr addrspace(4) %[[ID1]], align 4
  ; CHECK-SPV-IR: %[[PIPE_STORAGE_1_BC:[0-9]+]] = bitcast ptr addrspace(1) %[[PIPE_STORAGE_1]] to ptr addrspace(1)
  ; CHECK-SPV-IR: %[[WRITE_PIPE:[0-9]+]] = call spir_func target("spirv.Pipe", 1) @_Z39__spirv_CreatePipeFromPipeStorage_writePU3AS119__spirv_PipeStorage(ptr addrspace(1) %[[PIPE_STORAGE_1_BC]])
  ; CHECK-SPV-IR: %[[WRITE_PIPE_WRAPPER:[0-9]+]] = addrspacecast ptr %mywpipe to ptr addrspace(4)
  ; CHECK-SPV-IR: call spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE1EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE1EEE(ptr addrspace(4) nocapture %[[WRITE_PIPE_WRAPPER]], target("spirv.Pipe", 1) %[[WRITE_PIPE]])

  ; CHECK-SPIRV: Load {{[0-9]+}} [[PIPE_STORAGE_ID0:[0-9]+]] [[SPIRV1]] 2 4
  ; CHECK-SPIRV: Bitcast {{[0-9]+}} [[PIPE_STORAGE_ID0_BC:[0-9]+]] [[PIPE_STORAGE_ID0]]
  ; CHECK-SPIRV: CreatePipeFromPipeStorage [[WRITE_PIPE]] [[WRITE_PIPE_ID:[0-9]+]] [[PIPE_STORAGE_ID0_BC]]
  ; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[GENERIC_WRITE_PIPE_WRAPPER_ID:[0-9]+]] [[WRITE_PIPE_WRAPPER_ID]]
  ; CHECK-SPIRV: FunctionCall {{[0-9]+}} {{[0-9]+}} [[WRITE_PIPE_WRAPPER_CTOR]] [[GENERIC_WRITE_PIPE_WRAPPER_ID]] [[WRITE_PIPE_ID]]

  %2 = load ptr addrspace(1), ptr addrspace(4) %1, align 4
  %3 = tail call spir_func target("spirv.Pipe", 1) @_Z39__spirv_CreatePipeFromPipeStorage_writePU3AS1K19__spirv_PipeStorage(ptr addrspace(1) %2)
  %4 = addrspacecast ptr %mywpipe to ptr addrspace(4)
  call spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE1EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE1EEE(ptr addrspace(4) %4, target("spirv.Pipe", 1) %3)


  ; CHECK-SPV-IR: %[[PIPE_STORAGE_2:[0-9]+]] = load ptr addrspace(1), ptr addrspace(4) %[[ID1]], align 4
  ; CHECK-SPV-IR: %[[PIPE_STORAGE_2_BC:[0-9]+]] = bitcast ptr addrspace(1) %[[PIPE_STORAGE_2]] to ptr addrspace(1)
  ; CHECK-SPV-IR: %[[READ_PIPE:[0-9]+]] = call spir_func target("spirv.Pipe", 0) @_Z38__spirv_CreatePipeFromPipeStorage_readPU3AS119__spirv_PipeStorage(ptr addrspace(1) %[[PIPE_STORAGE_2_BC]])
  ; CHECK-SPV-IR: %[[READ_PIPE_WRAPPER:[0-9]+]] = addrspacecast ptr %myrpipe to ptr addrspace(4)
  ; CHECK-SPV-IR: call spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE0EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE0EEE(ptr addrspace(4) nocapture %[[READ_PIPE_WRAPPER]], target("spirv.Pipe", 0) %[[READ_PIPE]])

  ; CHECK-SPIRV: Load {{[0-9]+}} [[PIPE_STORAGE_ID1:[0-9]+]] [[SPIRV1]] 2 4
  ; CHECK-SPIRV: Bitcast {{[0-9]+}} [[PIPE_STORAGE_ID1_BC:[0-9]+]] [[PIPE_STORAGE_ID1]]
  ; CHECK-SPIRV: CreatePipeFromPipeStorage [[READ_PIPE]] [[READ_PIPE_ID:[0-9]+]] [[PIPE_STORAGE_ID1_BC]]
  ; CHECK-SPIRV: PtrCastToGeneric {{[0-9]+}} [[GENERIC_READ_PIPE_WRAPPER_ID:[0-9]+]] [[READ_PIPE_WRAPPER_ID]]
  ; CHECK-SPIRV: FunctionCall {{[0-9]+}} {{[0-9]+}} [[READ_PIPE_WRAPPER_CTOR]] [[GENERIC_READ_PIPE_WRAPPER_ID]] [[READ_PIPE_ID]]

  %5 = load ptr addrspace(1), ptr addrspace(4) %1, align 4
  %6 = tail call spir_func target("spirv.Pipe", 0) @_Z38__spirv_CreatePipeFromPipeStorage_readPU3AS1K19__spirv_PipeStorage(ptr addrspace(1) %5)
  %7 = addrspacecast ptr %myrpipe to ptr addrspace(4)
  call spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE0EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE0EEE(ptr addrspace(4) %7, target("spirv.Pipe", 0) %6)


  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE0EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE0EEE(ptr addrspace(4) nocapture %this, target("spirv.Pipe", 0) %handle) unnamed_addr align 2 {
entry:
  tail call spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE0EEC2EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE0EEE(ptr addrspace(4) %this, target("spirv.Pipe", 0) %handle)
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE0EEC2EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE0EEE(ptr addrspace(4) nocapture %this, target("spirv.Pipe", 0) %handle) unnamed_addr align 2 {
entry:
  %_handle = getelementptr inbounds %"class.cl::pipe<int __attribute__((ext_vector_type(4))), cl::pipe_access::read>", ptr addrspace(4) %this, i32 0, i32 0
  store target("spirv.Pipe", 0) %handle, target("spirv.Pipe", 0) addrspace(4)* %_handle, align 4, !tbaa !11
  ret void
}

; Function Attrs: nounwind
declare spir_func target("spirv.Pipe", 0) @_Z38__spirv_CreatePipeFromPipeStorage_readPU3AS1K19__spirv_PipeStorage(ptr addrspace(1))

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE1EEC1EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE1EEE(ptr addrspace(4) nocapture %this, target("spirv.Pipe", 1) %handle) unnamed_addr align 2 {
entry:
  tail call spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE1EEC2EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE1EEE(ptr addrspace(4) %this, target("spirv.Pipe", 1) %handle)
  ret void
}

; Function Attrs: nounwind
define linkonce_odr spir_func void @_ZNU3AS42cl4pipeIDv4_iLNS_11pipe_accessE1EEC2EPU3AS1NS_7__spirv10OpTypePipeILNS3_15AccessQualifierE1EEE(ptr addrspace(4) nocapture %this, target("spirv.Pipe", 1) %handle) unnamed_addr align 2 {
entry:
  %_handle = getelementptr inbounds %"class.cl::pipe<int __attribute__((ext_vector_type(4))), cl::pipe_access::write>", ptr addrspace(4) %this, i32 0, i32 0
  store target("spirv.Pipe", 1) %handle, target("spirv.Pipe", 1) addrspace(4)* %_handle, align 4, !tbaa !13
  ret void
}

; Function Attrs: nounwind
declare spir_func target("spirv.Pipe", 1) @_Z39__spirv_CreatePipeFromPipeStorage_writePU3AS1K19__spirv_PipeStorage(ptr addrspace(1))

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}
!spirv.Source = !{!4}
!spirv.String = !{}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 2}
!2 = !{}
!3 = !{!"clang version 3.6.1 "}
!4 = !{i32 4, i32 202000}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTSN2cl12pipe_storageIDv4_iLj1EEE", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !8, i64 0}
!12 = !{!"_ZTSN2cl4pipeIDv4_iLNS_11pipe_accessE0EEE", !8, i64 0}
!13 = !{!14, !8, i64 0}
!14 = !{!"_ZTSN2cl4pipeIDv4_iLNS_11pipe_accessE1EEE", !8, i64 0}
