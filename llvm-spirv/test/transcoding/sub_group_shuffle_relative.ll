;; #pragma OPENCL EXTENSION cl_khr_subgroup_shuffle_relative : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;; 
;; kernel void testShuffleRelativeChar(global char* dst)
;; {
;; 	char v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeUChar(global uchar* dst)
;; {
;; 	uchar v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeShort(global short* dst)
;; {
;; 	short v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeUShort(global ushort* dst)
;; {
;; 	ushort v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeInt(global int* dst)
;; {
;; 	int v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeUInt(global uint* dst)
;; {
;; 	uint v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeLong(global long* dst)
;; {
;; 	long v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeULong(global ulong* dst)
;; {
;; 	ulong v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeFloat(global float* dst)
;; {
;; 	float v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeHalf(global half* dst)
;; {
;; 	half v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }
;; 
;; kernel void testShuffleRelativeDouble(global double* dst)
;; {
;; 	double v = 0;
;;     dst[0] = sub_group_shuffle_up( v, 0 );
;;     dst[1] = sub_group_shuffle_down( v, 0 );
;; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: {{[0-9]*}} Capability GroupNonUniformShuffleRelative

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

; ModuleID = 'sub_group_shuffle_relative.cl'
source_filename = "sub_group_shuffle_relative.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeChar
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_shuffle_upcj(i8 0, i32 0)
; CHECK-LLVM: call spir_func i8 @_Z22sub_group_shuffle_downcj(i8 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeChar(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func signext i8 @_Z20sub_group_shuffle_upcj(i8 signext 0, i32 0) #2
  store i8 %2, ptr addrspace(1) %0, align 1, !tbaa !7
  %3 = tail call spir_func signext i8 @_Z22sub_group_shuffle_downcj(i8 signext 0, i32 0) #2
  %4 = getelementptr inbounds i8, ptr addrspace(1) %0, i64 1
  store i8 %3, ptr addrspace(1) %4, align 1, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z20sub_group_shuffle_upcj(i8 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z22sub_group_shuffle_downcj(i8 signext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeUChar
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_shuffle_upcj(i8 0, i32 0)
; CHECK-LLVM: call spir_func i8 @_Z22sub_group_shuffle_downcj(i8 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeUChar(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func zeroext i8 @_Z20sub_group_shuffle_uphj(i8 zeroext 0, i32 0) #2
  store i8 %2, ptr addrspace(1) %0, align 1, !tbaa !7
  %3 = tail call spir_func zeroext i8 @_Z22sub_group_shuffle_downhj(i8 zeroext 0, i32 0) #2
  %4 = getelementptr inbounds i8, ptr addrspace(1) %0, i64 1
  store i8 %3, ptr addrspace(1) %4, align 1, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z20sub_group_shuffle_uphj(i8 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z22sub_group_shuffle_downhj(i8 zeroext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeShort
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_shuffle_upsj(i16 0, i32 0)
; CHECK-LLVM: call spir_func i16 @_Z22sub_group_shuffle_downsj(i16 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeShort(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !11 !kernel_arg_base_type !11 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func signext i16 @_Z20sub_group_shuffle_upsj(i16 signext 0, i32 0) #2
  store i16 %2, ptr addrspace(1) %0, align 2, !tbaa !12
  %3 = tail call spir_func signext i16 @_Z22sub_group_shuffle_downsj(i16 signext 0, i32 0) #2
  %4 = getelementptr inbounds i16, ptr addrspace(1) %0, i64 1
  store i16 %3, ptr addrspace(1) %4, align 2, !tbaa !12
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z20sub_group_shuffle_upsj(i16 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z22sub_group_shuffle_downsj(i16 signext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeUShort
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_shuffle_upsj(i16 0, i32 0)
; CHECK-LLVM: call spir_func i16 @_Z22sub_group_shuffle_downsj(i16 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeUShort(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func zeroext i16 @_Z20sub_group_shuffle_uptj(i16 zeroext 0, i32 0) #2
  store i16 %2, ptr addrspace(1) %0, align 2, !tbaa !12
  %3 = tail call spir_func zeroext i16 @_Z22sub_group_shuffle_downtj(i16 zeroext 0, i32 0) #2
  %4 = getelementptr inbounds i16, ptr addrspace(1) %0, i64 1
  store i16 %3, ptr addrspace(1) %4, align 2, !tbaa !12
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z20sub_group_shuffle_uptj(i16 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z22sub_group_shuffle_downtj(i16 zeroext, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeInt
; CHECK-LLVM: call spir_func i32 @_Z20sub_group_shuffle_upij(i32 0, i32 0)
; CHECK-LLVM: call spir_func i32 @_Z22sub_group_shuffle_downij(i32 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeInt(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !15 !kernel_arg_base_type !15 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z20sub_group_shuffle_upij(i32 0, i32 0) #2
  store i32 %2, ptr addrspace(1) %0, align 4, !tbaa !16
  %3 = tail call spir_func i32 @_Z22sub_group_shuffle_downij(i32 0, i32 0) #2
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %3, ptr addrspace(1) %4, align 4, !tbaa !16
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z20sub_group_shuffle_upij(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z22sub_group_shuffle_downij(i32, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeUInt
; CHECK-LLVM: call spir_func i32 @_Z20sub_group_shuffle_upij(i32 0, i32 0)
; CHECK-LLVM: call spir_func i32 @_Z22sub_group_shuffle_downij(i32 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeUInt(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !18 !kernel_arg_base_type !18 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z20sub_group_shuffle_upjj(i32 0, i32 0) #2
  store i32 %2, ptr addrspace(1) %0, align 4, !tbaa !16
  %3 = tail call spir_func i32 @_Z22sub_group_shuffle_downjj(i32 0, i32 0) #2
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %3, ptr addrspace(1) %4, align 4, !tbaa !16
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z20sub_group_shuffle_upjj(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z22sub_group_shuffle_downjj(i32, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeLong
; CHECK-LLVM: call spir_func i64 @_Z20sub_group_shuffle_uplj(i64 0, i32 0)
; CHECK-LLVM: call spir_func i64 @_Z22sub_group_shuffle_downlj(i64 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeLong(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !19 !kernel_arg_base_type !19 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i64 @_Z20sub_group_shuffle_uplj(i64 0, i32 0) #2
  store i64 %2, ptr addrspace(1) %0, align 8, !tbaa !20
  %3 = tail call spir_func i64 @_Z22sub_group_shuffle_downlj(i64 0, i32 0) #2
  %4 = getelementptr inbounds i64, ptr addrspace(1) %0, i64 1
  store i64 %3, ptr addrspace(1) %4, align 8, !tbaa !20
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z20sub_group_shuffle_uplj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z22sub_group_shuffle_downlj(i64, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeULong
; CHECK-LLVM: call spir_func i64 @_Z20sub_group_shuffle_uplj(i64 0, i32 0)
; CHECK-LLVM: call spir_func i64 @_Z22sub_group_shuffle_downlj(i64 0, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeULong(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !22 !kernel_arg_base_type !22 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i64 @_Z20sub_group_shuffle_upmj(i64 0, i32 0) #2
  store i64 %2, ptr addrspace(1) %0, align 8, !tbaa !20
  %3 = tail call spir_func i64 @_Z22sub_group_shuffle_downmj(i64 0, i32 0) #2
  %4 = getelementptr inbounds i64, ptr addrspace(1) %0, i64 1
  store i64 %3, ptr addrspace(1) %4, align 8, !tbaa !20
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z20sub_group_shuffle_upmj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z22sub_group_shuffle_downmj(i64, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeFloat
; CHECK-LLVM: call spir_func float @_Z20sub_group_shuffle_upfj(float 0.000000e+00, i32 0)
; CHECK-LLVM: call spir_func float @_Z22sub_group_shuffle_downfj(float 0.000000e+00, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeFloat(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !23 !kernel_arg_base_type !23 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func float @_Z20sub_group_shuffle_upfj(float 0.000000e+00, i32 0) #2
  store float %2, ptr addrspace(1) %0, align 4, !tbaa !24
  %3 = tail call spir_func float @_Z22sub_group_shuffle_downfj(float 0.000000e+00, i32 0) #2
  %4 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  store float %3, ptr addrspace(1) %4, align 4, !tbaa !24
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z20sub_group_shuffle_upfj(float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func float @_Z22sub_group_shuffle_downfj(float, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeHalf
; CHECK-LLVM: call spir_func half @_Z20sub_group_shuffle_upDhj(half 0xH0000, i32 0)
; CHECK-LLVM: call spir_func half @_Z22sub_group_shuffle_downDhj(half 0xH0000, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeHalf(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !26 !kernel_arg_base_type !26 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func half @_Z20sub_group_shuffle_upDhj(half 0xH0000, i32 0) #2
  store half %2, ptr addrspace(1) %0, align 2, !tbaa !27
  %3 = tail call spir_func half @_Z22sub_group_shuffle_downDhj(half 0xH0000, i32 0) #2
  %4 = getelementptr inbounds half, ptr addrspace(1) %0, i64 1
  store half %3, ptr addrspace(1) %4, align 2, !tbaa !27
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func half @_Z20sub_group_shuffle_upDhj(half, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func half @_Z22sub_group_shuffle_downDhj(half, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformShuffleUp [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformShuffleDown [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testShuffleRelativeDouble
; CHECK-LLVM: call spir_func double @_Z20sub_group_shuffle_updj(double 0.000000e+00, i32 0)
; CHECK-LLVM: call spir_func double @_Z22sub_group_shuffle_downdj(double 0.000000e+00, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testShuffleRelativeDouble(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !29 !kernel_arg_base_type !29 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func double @_Z20sub_group_shuffle_updj(double 0.000000e+00, i32 0) #2
  store double %2, ptr addrspace(1) %0, align 8, !tbaa !30
  %3 = tail call spir_func double @_Z22sub_group_shuffle_downdj(double 0.000000e+00, i32 0) #2
  %4 = getelementptr inbounds double, ptr addrspace(1) %0, i64 1
  store double %3, ptr addrspace(1) %4, align 8, !tbaa !30
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z20sub_group_shuffle_updj(double, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func double @_Z22sub_group_shuffle_downdj(double, i32) local_unnamed_addr #1

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
