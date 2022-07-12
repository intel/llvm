;; #pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;; 
;; kernel void testShuffleChar(global char* dst)
;; {
;; 	char v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleUChar(global uchar* dst)
;; {
;; 	uchar v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleShort(global short* dst)
;; {
;; 	short v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleUShort(global ushort* dst)
;; {
;; 	ushort v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleInt(global int* dst)
;; {
;; 	int v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleUInt(global uint* dst)
;; {
;; 	uint v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleLong(global long* dst)
;; {
;; 	long v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleULong(global ulong* dst)
;; {
;; 	ulong v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleFloat(global float* dst)
;; {
;; 	float v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleHalf(global half* dst)
;; {
;; 	half v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }
;; 
;; kernel void testShuffleDouble(global double* dst)
;; {
;; 	double v = 0;
;;     dst[0] = sub_group_shuffle( v, 0 );
;;     dst[1] = sub_group_shuffle_xor( v, 0 );
;; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-LLVM
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPV-IR

; CHECK-SPIRV-DAG: {{[0-9]*}} Capability GroupNonUniformShuffle

; CHECK-SPIRV-DAG: TypeInt   [[char:[0-9]+]]   8  0
; CHECK-SPIRV-DAG: TypeInt   [[short:[0-9]+]]  16 0
; CHECK-SPIRV-DAG: TypeInt   [[int:[0-9]+]]    32 0
; CHECK-SPIRV-DAG: TypeInt   [[long:[0-9]+]]   64 0
; CHECK-SPIRV-DAG: TypeFloat [[half:[0-9]+]]   16
; CHECK-SPIRV-DAG: TypeFloat [[float:[0-9]+]]  32
; CHECK-SPIRV-DAG: TypeFloat [[double:[0-9]+]] 64

; CHECK-SPIRV-DAG: Constant [[int]]    [[ScopeSubgroup:[0-9]+]] 3
; CHECK-SPIRV-DAG: Constant [[char]]   [[char_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[short]]  [[short_0:[0-9]+]]       0
; CHECK-SPIRV-DAG: Constant [[int]]    [[int_0:[0-9]+]]         0
; CHECK-SPIRV-DAG: Constant [[long]]   [[long_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[half]]   [[half_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[float]]  [[float_0:[0-9]+]]       0
; CHECK-SPIRV-DAG: Constant [[double]] [[double_0:[0-9]+]]      0

; ModuleID = 'sub_group_shuffle.cl'
source_filename = "sub_group_shuffle.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleChar

; CHECK-LLVM: call spir_func i8 @_Z17sub_group_shufflecj(i8 0, i32 0)
; CHECK-LLVM: call spir_func i8 @_Z21sub_group_shuffle_xorcj(i8 0, i32 0)

; CHECK-SPV-IR: call spir_func i8 @_Z30__spirv_GroupNonUniformShuffleicj(i32 3, i8 0, i32 0)
; CHECK-SPV-IR: call spir_func i8 @_Z33__spirv_GroupNonUniformShuffleXoricj(i32 3, i8 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleChar(i8 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func signext i8 @_Z17sub_group_shufflecj(i8 signext 0, i32 0) #2
  store i8 %2, i8 addrspace(1)* %0, align 1, !tbaa !7
  %3 = tail call spir_func signext i8 @_Z21sub_group_shuffle_xorcj(i8 signext 0, i32 0) #2
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z17sub_group_shufflecj(i8 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z21sub_group_shuffle_xorcj(i8 signext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleUChar

; CHECK-LLVM: call spir_func i8 @_Z17sub_group_shufflecj(i8 0, i32 0)
; CHECK-LLVM: call spir_func i8 @_Z21sub_group_shuffle_xorcj(i8 0, i32 0)

; CHECK-SPV-IR: call spir_func i8 @_Z30__spirv_GroupNonUniformShuffleicj(i32 3, i8 0, i32 0)
; CHECK-SPV-IR: call spir_func i8 @_Z33__spirv_GroupNonUniformShuffleXoricj(i32 3, i8 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleUChar(i8 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func zeroext i8 @_Z17sub_group_shufflehj(i8 zeroext 0, i32 0) #2
  store i8 %2, i8 addrspace(1)* %0, align 1, !tbaa !7
  %3 = tail call spir_func zeroext i8 @_Z21sub_group_shuffle_xorhj(i8 zeroext 0, i32 0) #2
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z17sub_group_shufflehj(i8 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z21sub_group_shuffle_xorhj(i8 zeroext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleShort

; CHECK-LLVM: call spir_func i16 @_Z17sub_group_shufflesj(i16 0, i32 0)
; CHECK-LLVM: call spir_func i16 @_Z21sub_group_shuffle_xorsj(i16 0, i32 0)

; CHECK-SPV-IR: call spir_func i16 @_Z30__spirv_GroupNonUniformShuffleisj(i32 3, i16 0, i32 0)
; CHECK-SPV-IR: call spir_func i16 @_Z33__spirv_GroupNonUniformShuffleXorisj(i32 3, i16 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleShort(i16 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !11 !kernel_arg_base_type !11 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func signext i16 @_Z17sub_group_shufflesj(i16 signext 0, i32 0) #2
  store i16 %2, i16 addrspace(1)* %0, align 2, !tbaa !12
  %3 = tail call spir_func signext i16 @_Z21sub_group_shuffle_xorsj(i16 signext 0, i32 0) #2
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2, !tbaa !12
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z17sub_group_shufflesj(i16 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z21sub_group_shuffle_xorsj(i16 signext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleUShort

; CHECK-LLVM: call spir_func i16 @_Z17sub_group_shufflesj(i16 0, i32 0)
; CHECK-LLVM: call spir_func i16 @_Z21sub_group_shuffle_xorsj(i16 0, i32 0)

; CHECK-SPV-IR: call spir_func i16 @_Z30__spirv_GroupNonUniformShuffleisj(i32 3, i16 0, i32 0)
; CHECK-SPV-IR: call spir_func i16 @_Z33__spirv_GroupNonUniformShuffleXorisj(i32 3, i16 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleUShort(i16 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func zeroext i16 @_Z17sub_group_shuffletj(i16 zeroext 0, i32 0) #2
  store i16 %2, i16 addrspace(1)* %0, align 2, !tbaa !12
  %3 = tail call spir_func zeroext i16 @_Z21sub_group_shuffle_xortj(i16 zeroext 0, i32 0) #2
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2, !tbaa !12
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z17sub_group_shuffletj(i16 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z21sub_group_shuffle_xortj(i16 zeroext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleInt

; CHECK-LLVM: call spir_func i32 @_Z17sub_group_shuffleij(i32 0, i32 0)
; CHECK-LLVM: call spir_func i32 @_Z21sub_group_shuffle_xorij(i32 0, i32 0)

; CHECK-SPV-IR: call spir_func i32 @_Z30__spirv_GroupNonUniformShuffleiij(i32 3, i32 0, i32 0)
; CHECK-SPV-IR: call spir_func i32 @_Z33__spirv_GroupNonUniformShuffleXoriij(i32 3, i32 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleInt(i32 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !15 !kernel_arg_base_type !15 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z17sub_group_shuffleij(i32 0, i32 0) #2
  store i32 %2, i32 addrspace(1)* %0, align 4, !tbaa !16
  %3 = tail call spir_func i32 @_Z21sub_group_shuffle_xorij(i32 0, i32 0) #2
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4, !tbaa !16
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z17sub_group_shuffleij(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z21sub_group_shuffle_xorij(i32, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleUInt

; CHECK-LLVM: call spir_func i32 @_Z17sub_group_shuffleij(i32 0, i32 0)
; CHECK-LLVM: call spir_func i32 @_Z21sub_group_shuffle_xorij(i32 0, i32 0)

; CHECK-SPV-IR: call spir_func i32 @_Z30__spirv_GroupNonUniformShuffleiij(i32 3, i32 0, i32 0)
; CHECK-SPV-IR: call spir_func i32 @_Z33__spirv_GroupNonUniformShuffleXoriij(i32 3, i32 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleUInt(i32 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !18 !kernel_arg_base_type !18 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z17sub_group_shufflejj(i32 0, i32 0) #2
  store i32 %2, i32 addrspace(1)* %0, align 4, !tbaa !16
  %3 = tail call spir_func i32 @_Z21sub_group_shuffle_xorjj(i32 0, i32 0) #2
  %4 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %3, i32 addrspace(1)* %4, align 4, !tbaa !16
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z17sub_group_shufflejj(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z21sub_group_shuffle_xorjj(i32, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleLong

; CHECK-LLVM: call spir_func i64 @_Z17sub_group_shufflelj(i64 0, i32 0)
; CHECK-LLVM: call spir_func i64 @_Z21sub_group_shuffle_xorlj(i64 0, i32 0)

; CHECK-SPV-IR: call spir_func i64 @_Z30__spirv_GroupNonUniformShuffleilj(i32 3, i64 0, i32 0)
; CHECK-SPV-IR: call spir_func i64 @_Z33__spirv_GroupNonUniformShuffleXorilj(i32 3, i64 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleLong(i64 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !19 !kernel_arg_base_type !19 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i64 @_Z17sub_group_shufflelj(i64 0, i32 0) #2
  store i64 %2, i64 addrspace(1)* %0, align 8, !tbaa !20
  %3 = tail call spir_func i64 @_Z21sub_group_shuffle_xorlj(i64 0, i32 0) #2
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8, !tbaa !20
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z17sub_group_shufflelj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z21sub_group_shuffle_xorlj(i64, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleULong

; CHECK-LLVM: call spir_func i64 @_Z17sub_group_shufflelj(i64 0, i32 0)
; CHECK-LLVM: call spir_func i64 @_Z21sub_group_shuffle_xorlj(i64 0, i32 0)

; CHECK-SPV-IR: call spir_func i64 @_Z30__spirv_GroupNonUniformShuffleilj(i32 3, i64 0, i32 0)
; CHECK-SPV-IR: call spir_func i64 @_Z33__spirv_GroupNonUniformShuffleXorilj(i32 3, i64 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleULong(i64 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !22 !kernel_arg_base_type !22 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i64 @_Z17sub_group_shufflemj(i64 0, i32 0) #2
  store i64 %2, i64 addrspace(1)* %0, align 8, !tbaa !20
  %3 = tail call spir_func i64 @_Z21sub_group_shuffle_xormj(i64 0, i32 0) #2
  %4 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
  store i64 %3, i64 addrspace(1)* %4, align 8, !tbaa !20
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z17sub_group_shufflemj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z21sub_group_shuffle_xormj(i64, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleFloat

; CHECK-LLVM: call spir_func float @_Z17sub_group_shufflefj(float 0.000000e+00, i32 0)
; CHECK-LLVM: call spir_func float @_Z21sub_group_shuffle_xorfj(float 0.000000e+00, i32 0)

; CHECK-SPV-IR: call spir_func float @_Z30__spirv_GroupNonUniformShuffleifj(i32 3, float 0.000000e+00, i32 0)
; CHECK-SPV-IR: call spir_func float @_Z33__spirv_GroupNonUniformShuffleXorifj(i32 3, float 0.000000e+00, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleFloat(float addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !23 !kernel_arg_base_type !23 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func float @_Z17sub_group_shufflefj(float 0.000000e+00, i32 0) #2
  store float %2, float addrspace(1)* %0, align 4, !tbaa !24
  %3 = tail call spir_func float @_Z21sub_group_shuffle_xorfj(float 0.000000e+00, i32 0) #2
  %4 = getelementptr inbounds float, float addrspace(1)* %0, i64 1
  store float %3, float addrspace(1)* %4, align 4, !tbaa !24
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z17sub_group_shufflefj(float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func float @_Z21sub_group_shuffle_xorfj(float, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleHalf

; CHECK-LLVM: call spir_func half @_Z17sub_group_shuffleDhj(half 0xH0000, i32 0)
; CHECK-LLVM: call spir_func half @_Z21sub_group_shuffle_xorDhj(half 0xH0000, i32 0)

; CHECK-SPV-IR: call spir_func half @_Z30__spirv_GroupNonUniformShuffleiDhj(i32 3, half 0xH0000, i32 0)
; CHECK-SPV-IR: call spir_func half @_Z33__spirv_GroupNonUniformShuffleXoriDhj(i32 3, half 0xH0000, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleHalf(half addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !26 !kernel_arg_base_type !26 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func half @_Z17sub_group_shuffleDhj(half 0xH0000, i32 0) #2
  store half %2, half addrspace(1)* %0, align 2, !tbaa !27
  %3 = tail call spir_func half @_Z21sub_group_shuffle_xorDhj(half 0xH0000, i32 0) #2
  %4 = getelementptr inbounds half, half addrspace(1)* %0, i64 1
  store half %3, half addrspace(1)* %4, align 2, !tbaa !27
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func half @_Z17sub_group_shuffleDhj(half, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func half @_Z21sub_group_shuffle_xorDhj(half, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffle [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleXor [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testShuffleDouble

; CHECK-LLVM: call spir_func double @_Z17sub_group_shuffledj(double 0.000000e+00, i32 0)
; CHECK-LLVM: call spir_func double @_Z21sub_group_shuffle_xordj(double 0.000000e+00, i32 0)

; CHECK-SPV-IR: call spir_func double @_Z30__spirv_GroupNonUniformShuffleidj(i32 3, double 0.000000e+00, i32 0)
; CHECK-SPV-IR: call spir_func double @_Z33__spirv_GroupNonUniformShuffleXoridj(i32 3, double 0.000000e+00, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleDouble(double addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !29 !kernel_arg_base_type !29 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func double @_Z17sub_group_shuffledj(double 0.000000e+00, i32 0) #2
  store double %2, double addrspace(1)* %0, align 8, !tbaa !30
  %3 = tail call spir_func double @_Z21sub_group_shuffle_xordj(double 0.000000e+00, i32 0) #2
  %4 = getelementptr inbounds double, double addrspace(1)* %0, i64 1
  store double %3, double addrspace(1)* %4, align 8, !tbaa !30
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z17sub_group_shuffledj(double, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func double @_Z21sub_group_shuffle_xordj(double, i32) local_unnamed_addr #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 9.0.1 (https://github.com/llvm/llvm-project.git cb6d58d1dcf36a29ae5dd24ff891d6552f00bac7)"}
!3 = !{i32 1}
!4 = !{!"none"}
!5 = !{!"char*"}
!6 = !{!""}
!7 = !{!8, !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"uchar*"}
!11 = !{!"short*"}
!12 = !{!13, !13, i64 0}
!13 = !{!"short", !8, i64 0}
!14 = !{!"ushort*"}
!15 = !{!"int*"}
!16 = !{!17, !17, i64 0}
!17 = !{!"int", !8, i64 0}
!18 = !{!"uint*"}
!19 = !{!"long*"}
!20 = !{!21, !21, i64 0}
!21 = !{!"long", !8, i64 0}
!22 = !{!"ulong*"}
!23 = !{!"float*"}
!24 = !{!25, !25, i64 0}
!25 = !{!"float", !8, i64 0}
!26 = !{!"half*"}
!27 = !{!28, !28, i64 0}
!28 = !{!"half", !8, i64 0}
!29 = !{!"double*"}
!30 = !{!31, !31, i64 0}
!31 = !{!"double", !8, i64 0}


