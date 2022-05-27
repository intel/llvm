;; #pragma OPENCL EXTENSION cl_khr_subgroup_extended_types : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;; 
;; kernel void testBroadcastChar()
;; {
;;     char16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastUChar()
;; {
;;     uchar16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastShort()
;; {
;;     short16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastUShort()
;; {
;;     ushort16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastInt()
;; {
;;     int16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastUInt()
;; {
;;     uint16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastLong()
;; {
;;     long16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastULong()
;; {
;;     ulong16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastFloat()
;; {
;;     float16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastHalf()
;; {
;;     half16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testBroadcastDouble()
;; {
;;     double16 v = 0;
;;     v.s0 = sub_group_broadcast(v.s0, 0);
;;     v.s01 = sub_group_broadcast(v.s01, 0);
;;     v.s012 = sub_group_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_broadcast(v.s01234567, 0);
;;     v = sub_group_broadcast(v, 0);
;; }
;; 
;; kernel void testReduceScanChar(global char* dst)
;; {
;;     char v = 0;
;;     dst[0] = sub_group_reduce_add(v);
;;     dst[1] = sub_group_reduce_min(v);
;;     dst[2] = sub_group_reduce_max(v);
;;     dst[3] = sub_group_scan_inclusive_add(v);
;;     dst[4] = sub_group_scan_inclusive_min(v);
;;     dst[5] = sub_group_scan_inclusive_max(v);
;;     dst[6] = sub_group_scan_exclusive_add(v);
;;     dst[7] = sub_group_scan_exclusive_min(v);
;;     dst[8] = sub_group_scan_exclusive_max(v);
;; }
;; 
;; kernel void testReduceScanUChar(global uchar* dst)
;; {
;;     uchar v = 0;
;;     dst[0] = sub_group_reduce_add(v);
;;     dst[1] = sub_group_reduce_min(v);
;;     dst[2] = sub_group_reduce_max(v);
;;     dst[3] = sub_group_scan_inclusive_add(v);
;;     dst[4] = sub_group_scan_inclusive_min(v);
;;     dst[5] = sub_group_scan_inclusive_max(v);
;;     dst[6] = sub_group_scan_exclusive_add(v);
;;     dst[7] = sub_group_scan_exclusive_min(v);
;;     dst[8] = sub_group_scan_exclusive_max(v);
;; }
;; 
;; kernel void testReduceScanShort(global short* dst)
;; {
;;     short v = 0;
;;     dst[0] = sub_group_reduce_add(v);
;;     dst[1] = sub_group_reduce_min(v);
;;     dst[2] = sub_group_reduce_max(v);
;;     dst[3] = sub_group_scan_inclusive_add(v);
;;     dst[4] = sub_group_scan_inclusive_min(v);
;;     dst[5] = sub_group_scan_inclusive_max(v);
;;     dst[6] = sub_group_scan_exclusive_add(v);
;;     dst[7] = sub_group_scan_exclusive_min(v);
;;     dst[8] = sub_group_scan_exclusive_max(v);
;; }
;; 
;; kernel void testReduceScanUShort(global ushort* dst)
;; {
;;     ushort v = 0;
;;     dst[0] = sub_group_reduce_add(v);
;;     dst[1] = sub_group_reduce_min(v);
;;     dst[2] = sub_group_reduce_max(v);
;;     dst[3] = sub_group_scan_inclusive_add(v);
;;     dst[4] = sub_group_scan_inclusive_min(v);
;;     dst[5] = sub_group_scan_inclusive_max(v);
;;     dst[6] = sub_group_scan_exclusive_add(v);
;;     dst[7] = sub_group_scan_exclusive_min(v);
;;     dst[8] = sub_group_scan_exclusive_max(v);
;; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: TypeInt   [[char:[0-9]+]]   8  0
; CHECK-SPIRV-DAG: TypeInt   [[short:[0-9]+]]  16 0
; CHECK-SPIRV-DAG: TypeInt   [[int:[0-9]+]]    32 0
; CHECK-SPIRV-DAG: TypeInt   [[long:[0-9]+]]   64 0
; CHECK-SPIRV-DAG: TypeFloat [[half:[0-9]+]]   16
; CHECK-SPIRV-DAG: TypeFloat [[float:[0-9]+]]  32
; CHECK-SPIRV-DAG: TypeFloat [[double:[0-9]+]] 64

; CHECK-SPIRV-DAG: TypeVector [[char2:[0-9]+]]  [[char]] 2
; CHECK-SPIRV-DAG: TypeVector [[char3:[0-9]+]]  [[char]] 3
; CHECK-SPIRV-DAG: TypeVector [[char4:[0-9]+]]  [[char]] 4
; CHECK-SPIRV-DAG: TypeVector [[char8:[0-9]+]]  [[char]] 8
; CHECK-SPIRV-DAG: TypeVector [[char16:[0-9]+]] [[char]] 16

; CHECK-SPIRV-DAG: TypeVector [[short2:[0-9]+]]  [[short]] 2
; CHECK-SPIRV-DAG: TypeVector [[short3:[0-9]+]]  [[short]] 3
; CHECK-SPIRV-DAG: TypeVector [[short4:[0-9]+]]  [[short]] 4
; CHECK-SPIRV-DAG: TypeVector [[short8:[0-9]+]]  [[short]] 8
; CHECK-SPIRV-DAG: TypeVector [[short16:[0-9]+]] [[short]] 16

; CHECK-SPIRV-DAG: TypeVector [[int2:[0-9]+]]  [[int]] 2
; CHECK-SPIRV-DAG: TypeVector [[int3:[0-9]+]]  [[int]] 3
; CHECK-SPIRV-DAG: TypeVector [[int4:[0-9]+]]  [[int]] 4
; CHECK-SPIRV-DAG: TypeVector [[int8:[0-9]+]]  [[int]] 8
; CHECK-SPIRV-DAG: TypeVector [[int16:[0-9]+]] [[int]] 16

; CHECK-SPIRV-DAG: TypeVector [[long2:[0-9]+]]  [[long]] 2
; CHECK-SPIRV-DAG: TypeVector [[long3:[0-9]+]]  [[long]] 3
; CHECK-SPIRV-DAG: TypeVector [[long4:[0-9]+]]  [[long]] 4
; CHECK-SPIRV-DAG: TypeVector [[long8:[0-9]+]]  [[long]] 8
; CHECK-SPIRV-DAG: TypeVector [[long16:[0-9]+]] [[long]] 16

; CHECK-SPIRV-DAG: TypeVector [[float2:[0-9]+]]  [[float]] 2
; CHECK-SPIRV-DAG: TypeVector [[float3:[0-9]+]]  [[float]] 3
; CHECK-SPIRV-DAG: TypeVector [[float4:[0-9]+]]  [[float]] 4
; CHECK-SPIRV-DAG: TypeVector [[float8:[0-9]+]]  [[float]] 8
; CHECK-SPIRV-DAG: TypeVector [[float16:[0-9]+]] [[float]] 16

; CHECK-SPIRV-DAG: TypeVector [[half2:[0-9]+]]  [[half]] 2
; CHECK-SPIRV-DAG: TypeVector [[half3:[0-9]+]]  [[half]] 3
; CHECK-SPIRV-DAG: TypeVector [[half4:[0-9]+]]  [[half]] 4
; CHECK-SPIRV-DAG: TypeVector [[half8:[0-9]+]]  [[half]] 8
; CHECK-SPIRV-DAG: TypeVector [[half16:[0-9]+]] [[half]] 16

; CHECK-SPIRV-DAG: TypeVector [[double2:[0-9]+]]  [[double]] 2
; CHECK-SPIRV-DAG: TypeVector [[double3:[0-9]+]]  [[double]] 3
; CHECK-SPIRV-DAG: TypeVector [[double4:[0-9]+]]  [[double]] 4
; CHECK-SPIRV-DAG: TypeVector [[double8:[0-9]+]]  [[double]] 8
; CHECK-SPIRV-DAG: TypeVector [[double16:[0-9]+]] [[double]] 16

; CHECK-SPIRV-DAG: Constant [[int]]    [[ScopeSubgroup:[0-9]+]] 3
; CHECK-SPIRV-DAG: Constant [[char]]   [[char_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[short]]  [[short_0:[0-9]+]]       0
; CHECK-SPIRV-DAG: Constant [[int]]    [[int_0:[0-9]+]]         0
; CHECK-SPIRV-DAG: Constant [[long]]   [[long_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[half]]   [[half_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[float]]  [[float_0:[0-9]+]]       0
; CHECK-SPIRV-DAG: Constant [[double]] [[double_0:[0-9]+]]      0

; ModuleID = 'sub_group_extended_types.cl'
source_filename = "sub_group_extended_types.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char2]] [[char2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char2]] {{[0-9]+}} [[ScopeSubgroup]] [[char2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char3]] [[char3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char3]] {{[0-9]+}} [[ScopeSubgroup]] [[char3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char4]] [[char4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char4]] {{[0-9]+}} [[ScopeSubgroup]] [[char4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char8]] [[char8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char8]] {{[0-9]+}} [[ScopeSubgroup]] [[char8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[char16]] [[char16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char16]] {{[0-9]+}} [[ScopeSubgroup]] [[char16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastChar
; CHECK-LLVM: call spir_func i8 @_Z19sub_group_broadcasthj(i8 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i8> @_Z19sub_group_broadcastDv2_hj(<2 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i8> @_Z19sub_group_broadcastDv3_hj(<3 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i8> @_Z19sub_group_broadcastDv4_hj(<4 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i8> @_Z19sub_group_broadcastDv8_hj(<8 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i8> @_Z19sub_group_broadcastDv16_hj(<16 x i8> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastChar() local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func signext i8 @_Z19sub_group_broadcastcj(i8 signext 0, i32 0) #6
  %2 = insertelement <16 x i8> <i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %1, i64 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i8> @_Z19sub_group_broadcastDv2_cj(<2 x i8> %3, i32 0) #6
  %5 = shufflevector <2 x i8> %4, <2 x i8> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i8> %5, <16 x i8> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i8> @_Z19sub_group_broadcastDv3_cj(<3 x i8> %7, i32 0) #6
  %9 = shufflevector <3 x i8> %8, <3 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i8> %9, <16 x i8> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i8> %10, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i8> @_Z19sub_group_broadcastDv4_cj(<4 x i8> %11, i32 0) #6
  %13 = shufflevector <4 x i8> %12, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i8> %13, <16 x i8> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i8> %14, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i8> @_Z19sub_group_broadcastDv8_cj(<8 x i8> %15, i32 0) #6
  %17 = shufflevector <8 x i8> %16, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i8> %17, <16 x i8> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i8> @_Z19sub_group_broadcastDv16_cj(<16 x i8> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z19sub_group_broadcastcj(i8 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i8> @_Z19sub_group_broadcastDv2_cj(<2 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i8> @_Z19sub_group_broadcastDv3_cj(<3 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i8> @_Z19sub_group_broadcastDv4_cj(<4 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i8> @_Z19sub_group_broadcastDv8_cj(<8 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i8> @_Z19sub_group_broadcastDv16_cj(<16 x i8>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char2]] [[char2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char2]] {{[0-9]+}} [[ScopeSubgroup]] [[char2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char3]] [[char3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char3]] {{[0-9]+}} [[ScopeSubgroup]] [[char3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char4]] [[char4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char4]] {{[0-9]+}} [[ScopeSubgroup]] [[char4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char8]] [[char8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char8]] {{[0-9]+}} [[ScopeSubgroup]] [[char8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[char16]] [[char16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[char16]] {{[0-9]+}} [[ScopeSubgroup]] [[char16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastUChar
; CHECK-LLVM: call spir_func i8 @_Z19sub_group_broadcasthj(i8 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i8> @_Z19sub_group_broadcastDv2_hj(<2 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i8> @_Z19sub_group_broadcastDv3_hj(<3 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i8> @_Z19sub_group_broadcastDv4_hj(<4 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i8> @_Z19sub_group_broadcastDv8_hj(<8 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i8> @_Z19sub_group_broadcastDv16_hj(<16 x i8> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastUChar() local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func zeroext i8 @_Z19sub_group_broadcasthj(i8 zeroext 0, i32 0) #6
  %2 = insertelement <16 x i8> <i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %1, i64 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i8> @_Z19sub_group_broadcastDv2_hj(<2 x i8> %3, i32 0) #6
  %5 = shufflevector <2 x i8> %4, <2 x i8> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i8> %5, <16 x i8> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i8> @_Z19sub_group_broadcastDv3_hj(<3 x i8> %7, i32 0) #6
  %9 = shufflevector <3 x i8> %8, <3 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i8> %9, <16 x i8> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i8> %10, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i8> @_Z19sub_group_broadcastDv4_hj(<4 x i8> %11, i32 0) #6
  %13 = shufflevector <4 x i8> %12, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i8> %13, <16 x i8> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i8> %14, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i8> @_Z19sub_group_broadcastDv8_hj(<8 x i8> %15, i32 0) #6
  %17 = shufflevector <8 x i8> %16, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i8> %17, <16 x i8> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i8> @_Z19sub_group_broadcastDv16_hj(<16 x i8> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z19sub_group_broadcasthj(i8 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i8> @_Z19sub_group_broadcastDv2_hj(<2 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i8> @_Z19sub_group_broadcastDv3_hj(<3 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i8> @_Z19sub_group_broadcastDv4_hj(<4 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i8> @_Z19sub_group_broadcastDv8_hj(<8 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i8> @_Z19sub_group_broadcastDv16_hj(<16 x i8>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short2]] [[short2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short2]] {{[0-9]+}} [[ScopeSubgroup]] [[short2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short3]] [[short3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short3]] {{[0-9]+}} [[ScopeSubgroup]] [[short3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short4]] [[short4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short4]] {{[0-9]+}} [[ScopeSubgroup]] [[short4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short8]] [[short8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short8]] {{[0-9]+}} [[ScopeSubgroup]] [[short8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[short16]] [[short16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short16]] {{[0-9]+}} [[ScopeSubgroup]] [[short16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastShort
; CHECK-LLVM: call spir_func i16 @_Z19sub_group_broadcasttj(i16 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i16> @_Z19sub_group_broadcastDv2_tj(<2 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i16> @_Z19sub_group_broadcastDv3_tj(<3 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i16> @_Z19sub_group_broadcastDv4_tj(<4 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i16> @_Z19sub_group_broadcastDv8_tj(<8 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i16> @_Z19sub_group_broadcastDv16_tj(<16 x i16> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastShort() local_unnamed_addr #2 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func signext i16 @_Z19sub_group_broadcastsj(i16 signext 0, i32 0) #6
  %2 = insertelement <16 x i16> <i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %1, i64 0
  %3 = shufflevector <16 x i16> %2, <16 x i16> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i16> @_Z19sub_group_broadcastDv2_sj(<2 x i16> %3, i32 0) #6
  %5 = shufflevector <2 x i16> %4, <2 x i16> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i16> %5, <16 x i16> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i16> %6, <16 x i16> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i16> @_Z19sub_group_broadcastDv3_sj(<3 x i16> %7, i32 0) #6
  %9 = shufflevector <3 x i16> %8, <3 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i16> %9, <16 x i16> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i16> %10, <16 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i16> @_Z19sub_group_broadcastDv4_sj(<4 x i16> %11, i32 0) #6
  %13 = shufflevector <4 x i16> %12, <4 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i16> %13, <16 x i16> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i16> %14, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i16> @_Z19sub_group_broadcastDv8_sj(<8 x i16> %15, i32 0) #6
  %17 = shufflevector <8 x i16> %16, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i16> %17, <16 x i16> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i16> @_Z19sub_group_broadcastDv16_sj(<16 x i16> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z19sub_group_broadcastsj(i16 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i16> @_Z19sub_group_broadcastDv2_sj(<2 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i16> @_Z19sub_group_broadcastDv3_sj(<3 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i16> @_Z19sub_group_broadcastDv4_sj(<4 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i16> @_Z19sub_group_broadcastDv8_sj(<8 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i16> @_Z19sub_group_broadcastDv16_sj(<16 x i16>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short2]] [[short2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short2]] {{[0-9]+}} [[ScopeSubgroup]] [[short2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short3]] [[short3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short3]] {{[0-9]+}} [[ScopeSubgroup]] [[short3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short4]] [[short4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short4]] {{[0-9]+}} [[ScopeSubgroup]] [[short4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short8]] [[short8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short8]] {{[0-9]+}} [[ScopeSubgroup]] [[short8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[short16]] [[short16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[short16]] {{[0-9]+}} [[ScopeSubgroup]] [[short16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastUShort
; CHECK-LLVM: call spir_func i16 @_Z19sub_group_broadcasttj(i16 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i16> @_Z19sub_group_broadcastDv2_tj(<2 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i16> @_Z19sub_group_broadcastDv3_tj(<3 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i16> @_Z19sub_group_broadcastDv4_tj(<4 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i16> @_Z19sub_group_broadcastDv8_tj(<8 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i16> @_Z19sub_group_broadcastDv16_tj(<16 x i16> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastUShort() local_unnamed_addr #2 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func zeroext i16 @_Z19sub_group_broadcasttj(i16 zeroext 0, i32 0) #6
  %2 = insertelement <16 x i16> <i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %1, i64 0
  %3 = shufflevector <16 x i16> %2, <16 x i16> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i16> @_Z19sub_group_broadcastDv2_tj(<2 x i16> %3, i32 0) #6
  %5 = shufflevector <2 x i16> %4, <2 x i16> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i16> %5, <16 x i16> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i16> %6, <16 x i16> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i16> @_Z19sub_group_broadcastDv3_tj(<3 x i16> %7, i32 0) #6
  %9 = shufflevector <3 x i16> %8, <3 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i16> %9, <16 x i16> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i16> %10, <16 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i16> @_Z19sub_group_broadcastDv4_tj(<4 x i16> %11, i32 0) #6
  %13 = shufflevector <4 x i16> %12, <4 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i16> %13, <16 x i16> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i16> %14, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i16> @_Z19sub_group_broadcastDv8_tj(<8 x i16> %15, i32 0) #6
  %17 = shufflevector <8 x i16> %16, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i16> %17, <16 x i16> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i16> @_Z19sub_group_broadcastDv16_tj(<16 x i16> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z19sub_group_broadcasttj(i16 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i16> @_Z19sub_group_broadcastDv2_tj(<2 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i16> @_Z19sub_group_broadcastDv3_tj(<3 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i16> @_Z19sub_group_broadcastDv4_tj(<4 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i16> @_Z19sub_group_broadcastDv8_tj(<8 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i16> @_Z19sub_group_broadcastDv16_tj(<16 x i16>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int2]] [[int2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int2]] {{[0-9]+}} [[ScopeSubgroup]] [[int2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int3]] [[int3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int3]] {{[0-9]+}} [[ScopeSubgroup]] [[int3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int4]] [[int4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int4]] {{[0-9]+}} [[ScopeSubgroup]] [[int4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int8]] [[int8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int8]] {{[0-9]+}} [[ScopeSubgroup]] [[int8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[int16]] [[int16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int16]] {{[0-9]+}} [[ScopeSubgroup]] [[int16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastInt
; CHECK-LLVM: call spir_func i32 @_Z19sub_group_broadcastjj(i32 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i32> @_Z19sub_group_broadcastDv2_jj(<2 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i32> @_Z19sub_group_broadcastDv3_jj(<3 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i32> @_Z19sub_group_broadcastDv4_jj(<4 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i32> @_Z19sub_group_broadcastDv8_jj(<8 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i32> @_Z19sub_group_broadcastDv16_jj(<16 x i32> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastInt() local_unnamed_addr #3 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i32 @_Z19sub_group_broadcastij(i32 0, i32 0) #6
  %2 = insertelement <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %1, i64 0
  %3 = shufflevector <16 x i32> %2, <16 x i32> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i32> @_Z19sub_group_broadcastDv2_ij(<2 x i32> %3, i32 0) #6
  %5 = shufflevector <2 x i32> %4, <2 x i32> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i32> %5, <16 x i32> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i32> %6, <16 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i32> @_Z19sub_group_broadcastDv3_ij(<3 x i32> %7, i32 0) #6
  %9 = shufflevector <3 x i32> %8, <3 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i32> %9, <16 x i32> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i32> %10, <16 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i32> @_Z19sub_group_broadcastDv4_ij(<4 x i32> %11, i32 0) #6
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i32> %13, <16 x i32> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i32> %14, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i32> @_Z19sub_group_broadcastDv8_ij(<8 x i32> %15, i32 0) #6
  %17 = shufflevector <8 x i32> %16, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i32> %17, <16 x i32> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i32> @_Z19sub_group_broadcastDv16_ij(<16 x i32> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z19sub_group_broadcastij(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i32> @_Z19sub_group_broadcastDv2_ij(<2 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i32> @_Z19sub_group_broadcastDv3_ij(<3 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i32> @_Z19sub_group_broadcastDv4_ij(<4 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i32> @_Z19sub_group_broadcastDv8_ij(<8 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i32> @_Z19sub_group_broadcastDv16_ij(<16 x i32>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int2]] [[int2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int2]] {{[0-9]+}} [[ScopeSubgroup]] [[int2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int3]] [[int3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int3]] {{[0-9]+}} [[ScopeSubgroup]] [[int3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int4]] [[int4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int4]] {{[0-9]+}} [[ScopeSubgroup]] [[int4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int8]] [[int8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int8]] {{[0-9]+}} [[ScopeSubgroup]] [[int8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[int16]] [[int16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[int16]] {{[0-9]+}} [[ScopeSubgroup]] [[int16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastUInt
; CHECK-LLVM: call spir_func i32 @_Z19sub_group_broadcastjj(i32 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i32> @_Z19sub_group_broadcastDv2_jj(<2 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i32> @_Z19sub_group_broadcastDv3_jj(<3 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i32> @_Z19sub_group_broadcastDv4_jj(<4 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i32> @_Z19sub_group_broadcastDv8_jj(<8 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i32> @_Z19sub_group_broadcastDv16_jj(<16 x i32> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastUInt() local_unnamed_addr #3 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i32 @_Z19sub_group_broadcastjj(i32 0, i32 0) #6
  %2 = insertelement <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %1, i64 0
  %3 = shufflevector <16 x i32> %2, <16 x i32> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i32> @_Z19sub_group_broadcastDv2_jj(<2 x i32> %3, i32 0) #6
  %5 = shufflevector <2 x i32> %4, <2 x i32> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i32> %5, <16 x i32> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i32> %6, <16 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i32> @_Z19sub_group_broadcastDv3_jj(<3 x i32> %7, i32 0) #6
  %9 = shufflevector <3 x i32> %8, <3 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i32> %9, <16 x i32> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i32> %10, <16 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i32> @_Z19sub_group_broadcastDv4_jj(<4 x i32> %11, i32 0) #6
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i32> %13, <16 x i32> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i32> %14, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i32> @_Z19sub_group_broadcastDv8_jj(<8 x i32> %15, i32 0) #6
  %17 = shufflevector <8 x i32> %16, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i32> %17, <16 x i32> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i32> @_Z19sub_group_broadcastDv16_jj(<16 x i32> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z19sub_group_broadcastjj(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i32> @_Z19sub_group_broadcastDv2_jj(<2 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i32> @_Z19sub_group_broadcastDv3_jj(<3 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i32> @_Z19sub_group_broadcastDv4_jj(<4 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i32> @_Z19sub_group_broadcastDv8_jj(<8 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i32> @_Z19sub_group_broadcastDv16_jj(<16 x i32>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long2]] [[long2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long2]] {{[0-9]+}} [[ScopeSubgroup]] [[long2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long3]] [[long3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long3]] {{[0-9]+}} [[ScopeSubgroup]] [[long3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long4]] [[long4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long4]] {{[0-9]+}} [[ScopeSubgroup]] [[long4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long8]] [[long8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long8]] {{[0-9]+}} [[ScopeSubgroup]] [[long8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[long16]] [[long16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long16]] {{[0-9]+}} [[ScopeSubgroup]] [[long16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastLong
; CHECK-LLVM: call spir_func i64 @_Z19sub_group_broadcastmj(i64 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i64> @_Z19sub_group_broadcastDv2_mj(<2 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i64> @_Z19sub_group_broadcastDv3_mj(<3 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i64> @_Z19sub_group_broadcastDv4_mj(<4 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i64> @_Z19sub_group_broadcastDv8_mj(<8 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i64> @_Z19sub_group_broadcastDv16_mj(<16 x i64> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastLong() local_unnamed_addr #4 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i64 @_Z19sub_group_broadcastlj(i64 0, i32 0) #6
  %2 = insertelement <16 x i64> <i64 undef, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0>, i64 %1, i64 0
  %3 = shufflevector <16 x i64> %2, <16 x i64> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i64> @_Z19sub_group_broadcastDv2_lj(<2 x i64> %3, i32 0) #6
  %5 = shufflevector <2 x i64> %4, <2 x i64> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i64> %5, <16 x i64> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i64> %6, <16 x i64> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i64> @_Z19sub_group_broadcastDv3_lj(<3 x i64> %7, i32 0) #6
  %9 = shufflevector <3 x i64> %8, <3 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i64> %9, <16 x i64> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i64> %10, <16 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i64> @_Z19sub_group_broadcastDv4_lj(<4 x i64> %11, i32 0) #6
  %13 = shufflevector <4 x i64> %12, <4 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i64> %13, <16 x i64> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i64> %14, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i64> @_Z19sub_group_broadcastDv8_lj(<8 x i64> %15, i32 0) #6
  %17 = shufflevector <8 x i64> %16, <8 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i64> %17, <16 x i64> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i64> @_Z19sub_group_broadcastDv16_lj(<16 x i64> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z19sub_group_broadcastlj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i64> @_Z19sub_group_broadcastDv2_lj(<2 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i64> @_Z19sub_group_broadcastDv3_lj(<3 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i64> @_Z19sub_group_broadcastDv4_lj(<4 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i64> @_Z19sub_group_broadcastDv8_lj(<8 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i64> @_Z19sub_group_broadcastDv16_lj(<16 x i64>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long2]] [[long2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long2]] {{[0-9]+}} [[ScopeSubgroup]] [[long2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long3]] [[long3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long3]] {{[0-9]+}} [[ScopeSubgroup]] [[long3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long4]] [[long4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long4]] {{[0-9]+}} [[ScopeSubgroup]] [[long4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long8]] [[long8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long8]] {{[0-9]+}} [[ScopeSubgroup]] [[long8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[long16]] [[long16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[long16]] {{[0-9]+}} [[ScopeSubgroup]] [[long16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastULong
; CHECK-LLVM: call spir_func i64 @_Z19sub_group_broadcastmj(i64 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i64> @_Z19sub_group_broadcastDv2_mj(<2 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i64> @_Z19sub_group_broadcastDv3_mj(<3 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i64> @_Z19sub_group_broadcastDv4_mj(<4 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i64> @_Z19sub_group_broadcastDv8_mj(<8 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i64> @_Z19sub_group_broadcastDv16_mj(<16 x i64> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastULong() local_unnamed_addr #4 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i64 @_Z19sub_group_broadcastmj(i64 0, i32 0) #6
  %2 = insertelement <16 x i64> <i64 undef, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0>, i64 %1, i64 0
  %3 = shufflevector <16 x i64> %2, <16 x i64> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i64> @_Z19sub_group_broadcastDv2_mj(<2 x i64> %3, i32 0) #6
  %5 = shufflevector <2 x i64> %4, <2 x i64> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i64> %5, <16 x i64> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i64> %6, <16 x i64> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i64> @_Z19sub_group_broadcastDv3_mj(<3 x i64> %7, i32 0) #6
  %9 = shufflevector <3 x i64> %8, <3 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i64> %9, <16 x i64> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i64> %10, <16 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i64> @_Z19sub_group_broadcastDv4_mj(<4 x i64> %11, i32 0) #6
  %13 = shufflevector <4 x i64> %12, <4 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i64> %13, <16 x i64> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i64> %14, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i64> @_Z19sub_group_broadcastDv8_mj(<8 x i64> %15, i32 0) #6
  %17 = shufflevector <8 x i64> %16, <8 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i64> %17, <16 x i64> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i64> @_Z19sub_group_broadcastDv16_mj(<16 x i64> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z19sub_group_broadcastmj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i64> @_Z19sub_group_broadcastDv2_mj(<2 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i64> @_Z19sub_group_broadcastDv3_mj(<3 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i64> @_Z19sub_group_broadcastDv4_mj(<4 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i64> @_Z19sub_group_broadcastDv8_mj(<8 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i64> @_Z19sub_group_broadcastDv16_mj(<16 x i64>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float2]] [[float2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[float2]] {{[0-9]+}} [[ScopeSubgroup]] [[float2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float3]] [[float3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[float3]] {{[0-9]+}} [[ScopeSubgroup]] [[float3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float4]] [[float4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[float4]] {{[0-9]+}} [[ScopeSubgroup]] [[float4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float8]] [[float8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[float8]] {{[0-9]+}} [[ScopeSubgroup]] [[float8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[float16]] [[float16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[float16]] {{[0-9]+}} [[ScopeSubgroup]] [[float16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastFloat
; CHECK-LLVM: call spir_func float @_Z19sub_group_broadcastfj(float {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x float> @_Z19sub_group_broadcastDv2_fj(<2 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x float> @_Z19sub_group_broadcastDv3_fj(<3 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x float> @_Z19sub_group_broadcastDv4_fj(<4 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x float> @_Z19sub_group_broadcastDv8_fj(<8 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x float> @_Z19sub_group_broadcastDv16_fj(<16 x float> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastFloat() local_unnamed_addr #3 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func float @_Z19sub_group_broadcastfj(float 0.000000e+00, i32 0) #6
  %2 = insertelement <16 x float> <float undef, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %1, i64 0
  %3 = shufflevector <16 x float> %2, <16 x float> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x float> @_Z19sub_group_broadcastDv2_fj(<2 x float> %3, i32 0) #6
  %5 = shufflevector <2 x float> %4, <2 x float> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x float> %5, <16 x float> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x float> %6, <16 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x float> @_Z19sub_group_broadcastDv3_fj(<3 x float> %7, i32 0) #6
  %9 = shufflevector <3 x float> %8, <3 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x float> %9, <16 x float> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x float> %10, <16 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x float> @_Z19sub_group_broadcastDv4_fj(<4 x float> %11, i32 0) #6
  %13 = shufflevector <4 x float> %12, <4 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x float> %13, <16 x float> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x float> %14, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x float> @_Z19sub_group_broadcastDv8_fj(<8 x float> %15, i32 0) #6
  %17 = shufflevector <8 x float> %16, <8 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x float> %17, <16 x float> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x float> @_Z19sub_group_broadcastDv16_fj(<16 x float> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z19sub_group_broadcastfj(float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x float> @_Z19sub_group_broadcastDv2_fj(<2 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x float> @_Z19sub_group_broadcastDv3_fj(<3 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x float> @_Z19sub_group_broadcastDv4_fj(<4 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x float> @_Z19sub_group_broadcastDv8_fj(<8 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x float> @_Z19sub_group_broadcastDv16_fj(<16 x float>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half2]] [[half2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[half2]] {{[0-9]+}} [[ScopeSubgroup]] [[half2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half3]] [[half3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[half3]] {{[0-9]+}} [[ScopeSubgroup]] [[half3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half4]] [[half4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[half4]] {{[0-9]+}} [[ScopeSubgroup]] [[half4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half8]] [[half8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[half8]] {{[0-9]+}} [[ScopeSubgroup]] [[half8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[half16]] [[half16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[half16]] {{[0-9]+}} [[ScopeSubgroup]] [[half16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastHalf
; CHECK-LLVM: call spir_func half @_Z19sub_group_broadcastDhj(half {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x half> @_Z19sub_group_broadcastDv2_Dhj(<2 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x half> @_Z19sub_group_broadcastDv3_Dhj(<3 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x half> @_Z19sub_group_broadcastDv4_Dhj(<4 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x half> @_Z19sub_group_broadcastDv8_Dhj(<8 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x half> @_Z19sub_group_broadcastDv16_Dhj(<16 x half> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastHalf() local_unnamed_addr #2 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func half @_Z19sub_group_broadcastDhj(half 0xH0000, i32 0) #6
  %2 = insertelement <16 x half> <half undef, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000>, half %1, i64 0
  %3 = shufflevector <16 x half> %2, <16 x half> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x half> @_Z19sub_group_broadcastDv2_Dhj(<2 x half> %3, i32 0) #6
  %5 = shufflevector <2 x half> %4, <2 x half> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x half> %5, <16 x half> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x half> %6, <16 x half> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x half> @_Z19sub_group_broadcastDv3_Dhj(<3 x half> %7, i32 0) #6
  %9 = shufflevector <3 x half> %8, <3 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x half> %9, <16 x half> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x half> %10, <16 x half> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x half> @_Z19sub_group_broadcastDv4_Dhj(<4 x half> %11, i32 0) #6
  %13 = shufflevector <4 x half> %12, <4 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x half> %13, <16 x half> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x half> %14, <16 x half> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x half> @_Z19sub_group_broadcastDv8_Dhj(<8 x half> %15, i32 0) #6
  %17 = shufflevector <8 x half> %16, <8 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x half> %17, <16 x half> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x half> @_Z19sub_group_broadcastDv16_Dhj(<16 x half> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func half @_Z19sub_group_broadcastDhj(half, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x half> @_Z19sub_group_broadcastDv2_Dhj(<2 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x half> @_Z19sub_group_broadcastDv3_Dhj(<3 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x half> @_Z19sub_group_broadcastDv4_Dhj(<4 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x half> @_Z19sub_group_broadcastDv8_Dhj(<8 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x half> @_Z19sub_group_broadcastDv16_Dhj(<16 x half>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupBroadcast [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double2]] [[double2_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[double2]] {{[0-9]+}} [[ScopeSubgroup]] [[double2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double3]] [[double3_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[double3]] {{[0-9]+}} [[ScopeSubgroup]] [[double3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double4]] [[double4_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[double4]] {{[0-9]+}} [[ScopeSubgroup]] [[double4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double8]] [[double8_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[double8]] {{[0-9]+}} [[ScopeSubgroup]] [[double8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[double16]] [[double16_0:[0-9]+]] 
; CHECK-SPIRV: GroupBroadcast [[double16]] {{[0-9]+}} [[ScopeSubgroup]] [[double16_0]] [[int_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testBroadcastDouble
; CHECK-LLVM: call spir_func double @_Z19sub_group_broadcastdj(double {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x double> @_Z19sub_group_broadcastDv2_dj(<2 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x double> @_Z19sub_group_broadcastDv3_dj(<3 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x double> @_Z19sub_group_broadcastDv4_dj(<4 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x double> @_Z19sub_group_broadcastDv8_dj(<8 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x double> @_Z19sub_group_broadcastDv16_dj(<16 x double> {{.*}}, i32 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBroadcastDouble() local_unnamed_addr #4 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func double @_Z19sub_group_broadcastdj(double 0.000000e+00, i32 0) #6
  %2 = insertelement <16 x double> <double undef, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00>, double %1, i64 0
  %3 = shufflevector <16 x double> %2, <16 x double> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x double> @_Z19sub_group_broadcastDv2_dj(<2 x double> %3, i32 0) #6
  %5 = shufflevector <2 x double> %4, <2 x double> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x double> %5, <16 x double> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x double> %6, <16 x double> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x double> @_Z19sub_group_broadcastDv3_dj(<3 x double> %7, i32 0) #6
  %9 = shufflevector <3 x double> %8, <3 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x double> %9, <16 x double> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x double> %10, <16 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x double> @_Z19sub_group_broadcastDv4_dj(<4 x double> %11, i32 0) #6
  %13 = shufflevector <4 x double> %12, <4 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x double> %13, <16 x double> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x double> %14, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x double> @_Z19sub_group_broadcastDv8_dj(<8 x double> %15, i32 0) #6
  %17 = shufflevector <8 x double> %16, <8 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x double> %17, <16 x double> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x double> @_Z19sub_group_broadcastDv16_dj(<16 x double> %18, i32 0) #6
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z19sub_group_broadcastdj(double, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x double> @_Z19sub_group_broadcastDv2_dj(<2 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x double> @_Z19sub_group_broadcastDv3_dj(<3 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x double> @_Z19sub_group_broadcastDv4_dj(<4 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x double> @_Z19sub_group_broadcastDv8_dj(<8 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x double> @_Z19sub_group_broadcastDv16_dj(<16 x double>, i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupIAdd [[char]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[char_0]]
; CHECK-SPIRV: GroupSMin [[char]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[char_0]]
; CHECK-SPIRV: GroupSMax [[char]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[char_0]]
; CHECK-SPIRV: GroupIAdd [[char]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[char_0]]
; CHECK-SPIRV: GroupSMin [[char]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[char_0]]
; CHECK-SPIRV: GroupSMax [[char]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[char_0]]
; CHECK-SPIRV: GroupIAdd [[char]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[char_0]]
; CHECK-SPIRV: GroupSMin [[char]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[char_0]]
; CHECK-SPIRV: GroupSMax [[char]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[char_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testReduceScanChar
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_reduce_addc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_reduce_minc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_reduce_maxc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_inclusive_addc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_inclusive_minc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_inclusive_maxc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_exclusive_addc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_exclusive_minc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_exclusive_maxc(i8 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testReduceScanChar(i8 addrspace(1)* nocapture) local_unnamed_addr #5 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  %2 = tail call spir_func signext i8 @_Z20sub_group_reduce_addc(i8 signext 0) #6
  store i8 %2, i8 addrspace(1)* %0, align 1, !tbaa !8
  %3 = tail call spir_func signext i8 @_Z20sub_group_reduce_minc(i8 signext 0) #6
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1, !tbaa !8
  %5 = tail call spir_func signext i8 @_Z20sub_group_reduce_maxc(i8 signext 0) #6
  %6 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 2
  store i8 %5, i8 addrspace(1)* %6, align 1, !tbaa !8
  %7 = tail call spir_func signext i8 @_Z28sub_group_scan_inclusive_addc(i8 signext 0) #6
  %8 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 3
  store i8 %7, i8 addrspace(1)* %8, align 1, !tbaa !8
  %9 = tail call spir_func signext i8 @_Z28sub_group_scan_inclusive_minc(i8 signext 0) #6
  %10 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 4
  store i8 %9, i8 addrspace(1)* %10, align 1, !tbaa !8
  %11 = tail call spir_func signext i8 @_Z28sub_group_scan_inclusive_maxc(i8 signext 0) #6
  %12 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 5
  store i8 %11, i8 addrspace(1)* %12, align 1, !tbaa !8
  %13 = tail call spir_func signext i8 @_Z28sub_group_scan_exclusive_addc(i8 signext 0) #6
  %14 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 6
  store i8 %13, i8 addrspace(1)* %14, align 1, !tbaa !8
  %15 = tail call spir_func signext i8 @_Z28sub_group_scan_exclusive_minc(i8 signext 0) #6
  %16 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 7
  store i8 %15, i8 addrspace(1)* %16, align 1, !tbaa !8
  %17 = tail call spir_func signext i8 @_Z28sub_group_scan_exclusive_maxc(i8 signext 0) #6
  %18 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 8
  store i8 %17, i8 addrspace(1)* %18, align 1, !tbaa !8
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z20sub_group_reduce_addc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z20sub_group_reduce_minc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z20sub_group_reduce_maxc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z28sub_group_scan_inclusive_addc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z28sub_group_scan_inclusive_minc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z28sub_group_scan_inclusive_maxc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z28sub_group_scan_exclusive_addc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z28sub_group_scan_exclusive_minc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z28sub_group_scan_exclusive_maxc(i8 signext) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupIAdd [[char]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[char_0]]
; CHECK-SPIRV: GroupUMin [[char]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[char_0]]
; CHECK-SPIRV: GroupUMax [[char]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[char_0]]
; CHECK-SPIRV: GroupIAdd [[char]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[char_0]]
; CHECK-SPIRV: GroupUMin [[char]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[char_0]]
; CHECK-SPIRV: GroupUMax [[char]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[char_0]]
; CHECK-SPIRV: GroupIAdd [[char]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[char_0]]
; CHECK-SPIRV: GroupUMin [[char]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[char_0]]
; CHECK-SPIRV: GroupUMax [[char]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[char_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testReduceScanUChar
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_reduce_addc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_reduce_minh(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z20sub_group_reduce_maxh(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_inclusive_addc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_inclusive_minh(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_inclusive_maxh(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_exclusive_addc(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_exclusive_minh(i8 0)
; CHECK-LLVM: call spir_func i8 @_Z28sub_group_scan_exclusive_maxh(i8 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testReduceScanUChar(i8 addrspace(1)* nocapture) local_unnamed_addr #5 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !11 !kernel_arg_base_type !11 !kernel_arg_type_qual !7 {
  %2 = tail call spir_func zeroext i8 @_Z20sub_group_reduce_addh(i8 zeroext 0) #6
  store i8 %2, i8 addrspace(1)* %0, align 1, !tbaa !8
  %3 = tail call spir_func zeroext i8 @_Z20sub_group_reduce_minh(i8 zeroext 0) #6
  %4 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 1
  store i8 %3, i8 addrspace(1)* %4, align 1, !tbaa !8
  %5 = tail call spir_func zeroext i8 @_Z20sub_group_reduce_maxh(i8 zeroext 0) #6
  %6 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 2
  store i8 %5, i8 addrspace(1)* %6, align 1, !tbaa !8
  %7 = tail call spir_func zeroext i8 @_Z28sub_group_scan_inclusive_addh(i8 zeroext 0) #6
  %8 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 3
  store i8 %7, i8 addrspace(1)* %8, align 1, !tbaa !8
  %9 = tail call spir_func zeroext i8 @_Z28sub_group_scan_inclusive_minh(i8 zeroext 0) #6
  %10 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 4
  store i8 %9, i8 addrspace(1)* %10, align 1, !tbaa !8
  %11 = tail call spir_func zeroext i8 @_Z28sub_group_scan_inclusive_maxh(i8 zeroext 0) #6
  %12 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 5
  store i8 %11, i8 addrspace(1)* %12, align 1, !tbaa !8
  %13 = tail call spir_func zeroext i8 @_Z28sub_group_scan_exclusive_addh(i8 zeroext 0) #6
  %14 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 6
  store i8 %13, i8 addrspace(1)* %14, align 1, !tbaa !8
  %15 = tail call spir_func zeroext i8 @_Z28sub_group_scan_exclusive_minh(i8 zeroext 0) #6
  %16 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 7
  store i8 %15, i8 addrspace(1)* %16, align 1, !tbaa !8
  %17 = tail call spir_func zeroext i8 @_Z28sub_group_scan_exclusive_maxh(i8 zeroext 0) #6
  %18 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 8
  store i8 %17, i8 addrspace(1)* %18, align 1, !tbaa !8
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z20sub_group_reduce_addh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z20sub_group_reduce_minh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z20sub_group_reduce_maxh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z28sub_group_scan_inclusive_addh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z28sub_group_scan_inclusive_minh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z28sub_group_scan_inclusive_maxh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z28sub_group_scan_exclusive_addh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z28sub_group_scan_exclusive_minh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z28sub_group_scan_exclusive_maxh(i8 zeroext) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupIAdd [[short]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[short_0]]
; CHECK-SPIRV: GroupSMin [[short]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[short_0]]
; CHECK-SPIRV: GroupSMax [[short]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[short_0]]
; CHECK-SPIRV: GroupIAdd [[short]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[short_0]]
; CHECK-SPIRV: GroupSMin [[short]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[short_0]]
; CHECK-SPIRV: GroupSMax [[short]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[short_0]]
; CHECK-SPIRV: GroupIAdd [[short]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[short_0]]
; CHECK-SPIRV: GroupSMin [[short]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[short_0]]
; CHECK-SPIRV: GroupSMax [[short]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[short_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testReduceScanShort
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_reduce_adds(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_reduce_mins(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_reduce_maxs(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_inclusive_adds(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_inclusive_mins(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_inclusive_maxs(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_exclusive_adds(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_exclusive_mins(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_exclusive_maxs(i16 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testReduceScanShort(i16 addrspace(1)* nocapture) local_unnamed_addr #5 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !12 !kernel_arg_base_type !12 !kernel_arg_type_qual !7 {
  %2 = tail call spir_func signext i16 @_Z20sub_group_reduce_adds(i16 signext 0) #6
  store i16 %2, i16 addrspace(1)* %0, align 2, !tbaa !13
  %3 = tail call spir_func signext i16 @_Z20sub_group_reduce_mins(i16 signext 0) #6
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2, !tbaa !13
  %5 = tail call spir_func signext i16 @_Z20sub_group_reduce_maxs(i16 signext 0) #6
  %6 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 2
  store i16 %5, i16 addrspace(1)* %6, align 2, !tbaa !13
  %7 = tail call spir_func signext i16 @_Z28sub_group_scan_inclusive_adds(i16 signext 0) #6
  %8 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 3
  store i16 %7, i16 addrspace(1)* %8, align 2, !tbaa !13
  %9 = tail call spir_func signext i16 @_Z28sub_group_scan_inclusive_mins(i16 signext 0) #6
  %10 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 4
  store i16 %9, i16 addrspace(1)* %10, align 2, !tbaa !13
  %11 = tail call spir_func signext i16 @_Z28sub_group_scan_inclusive_maxs(i16 signext 0) #6
  %12 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 5
  store i16 %11, i16 addrspace(1)* %12, align 2, !tbaa !13
  %13 = tail call spir_func signext i16 @_Z28sub_group_scan_exclusive_adds(i16 signext 0) #6
  %14 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 6
  store i16 %13, i16 addrspace(1)* %14, align 2, !tbaa !13
  %15 = tail call spir_func signext i16 @_Z28sub_group_scan_exclusive_mins(i16 signext 0) #6
  %16 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 7
  store i16 %15, i16 addrspace(1)* %16, align 2, !tbaa !13
  %17 = tail call spir_func signext i16 @_Z28sub_group_scan_exclusive_maxs(i16 signext 0) #6
  %18 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 8
  store i16 %17, i16 addrspace(1)* %18, align 2, !tbaa !13
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z20sub_group_reduce_adds(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z20sub_group_reduce_mins(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z20sub_group_reduce_maxs(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z28sub_group_scan_inclusive_adds(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z28sub_group_scan_inclusive_mins(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z28sub_group_scan_inclusive_maxs(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z28sub_group_scan_exclusive_adds(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z28sub_group_scan_exclusive_mins(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z28sub_group_scan_exclusive_maxs(i16 signext) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupIAdd [[short]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[short_0]]
; CHECK-SPIRV: GroupUMin [[short]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[short_0]]
; CHECK-SPIRV: GroupUMax [[short]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[short_0]]
; CHECK-SPIRV: GroupIAdd [[short]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[short_0]]
; CHECK-SPIRV: GroupUMin [[short]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[short_0]]
; CHECK-SPIRV: GroupUMax [[short]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[short_0]]
; CHECK-SPIRV: GroupIAdd [[short]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[short_0]]
; CHECK-SPIRV: GroupUMin [[short]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[short_0]]
; CHECK-SPIRV: GroupUMax [[short]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[short_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testReduceScanUShort
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_reduce_adds(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_reduce_mint(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z20sub_group_reduce_maxt(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_inclusive_adds(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_inclusive_mint(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_inclusive_maxt(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_exclusive_adds(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_exclusive_mint(i16 0)
; CHECK-LLVM: call spir_func i16 @_Z28sub_group_scan_exclusive_maxt(i16 0)

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testReduceScanUShort(i16 addrspace(1)* nocapture) local_unnamed_addr #5 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !15 !kernel_arg_base_type !15 !kernel_arg_type_qual !7 {
  %2 = tail call spir_func zeroext i16 @_Z20sub_group_reduce_addt(i16 zeroext 0) #6
  store i16 %2, i16 addrspace(1)* %0, align 2, !tbaa !13
  %3 = tail call spir_func zeroext i16 @_Z20sub_group_reduce_mint(i16 zeroext 0) #6
  %4 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 1
  store i16 %3, i16 addrspace(1)* %4, align 2, !tbaa !13
  %5 = tail call spir_func zeroext i16 @_Z20sub_group_reduce_maxt(i16 zeroext 0) #6
  %6 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 2
  store i16 %5, i16 addrspace(1)* %6, align 2, !tbaa !13
  %7 = tail call spir_func zeroext i16 @_Z28sub_group_scan_inclusive_addt(i16 zeroext 0) #6
  %8 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 3
  store i16 %7, i16 addrspace(1)* %8, align 2, !tbaa !13
  %9 = tail call spir_func zeroext i16 @_Z28sub_group_scan_inclusive_mint(i16 zeroext 0) #6
  %10 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 4
  store i16 %9, i16 addrspace(1)* %10, align 2, !tbaa !13
  %11 = tail call spir_func zeroext i16 @_Z28sub_group_scan_inclusive_maxt(i16 zeroext 0) #6
  %12 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 5
  store i16 %11, i16 addrspace(1)* %12, align 2, !tbaa !13
  %13 = tail call spir_func zeroext i16 @_Z28sub_group_scan_exclusive_addt(i16 zeroext 0) #6
  %14 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 6
  store i16 %13, i16 addrspace(1)* %14, align 2, !tbaa !13
  %15 = tail call spir_func zeroext i16 @_Z28sub_group_scan_exclusive_mint(i16 zeroext 0) #6
  %16 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 7
  store i16 %15, i16 addrspace(1)* %16, align 2, !tbaa !13
  %17 = tail call spir_func zeroext i16 @_Z28sub_group_scan_exclusive_maxt(i16 zeroext 0) #6
  %18 = getelementptr inbounds i16, i16 addrspace(1)* %0, i64 8
  store i16 %17, i16 addrspace(1)* %18, align 2, !tbaa !13
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z20sub_group_reduce_addt(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z20sub_group_reduce_mint(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z20sub_group_reduce_maxt(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z28sub_group_scan_inclusive_addt(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z28sub_group_scan_inclusive_mint(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z28sub_group_scan_inclusive_maxt(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z28sub_group_scan_exclusive_addt(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z28sub_group_scan_exclusive_mint(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z28sub_group_scan_exclusive_maxt(i16 zeroext) local_unnamed_addr #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="256" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="1024" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 9.0.1 (https://github.com/llvm/llvm-project.git cb6d58d1dcf36a29ae5dd24ff891d6552f00bac7)"}
!3 = !{}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"char*"}
!7 = !{!""}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"uchar*"}
!12 = !{!"short*"}
!13 = !{!14, !14, i64 0}
!14 = !{!"short", !9, i64 0}
!15 = !{!"ushort*"}