;; #pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;; 
;; kernel void testNonUniformBroadcastChars()
;; {
;;     char16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastUChars()
;; {
;;     uchar16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastShorts()
;; {
;;     short16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastUShorts()
;; {
;;     ushort16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastInts()
;; {
;;     int16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastUInts()
;; {
;;     uint16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastLongs()
;; {
;;     long16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastULongs()
;; {
;;     ulong16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastFloats()
;; {
;;     float16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastHalfs()
;; {
;;     half16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testNonUniformBroadcastDoubles()
;; {
;;     double16 v = 0;
;;     v.s0 = sub_group_non_uniform_broadcast(v.s0, 0);
;;     v.s01 = sub_group_non_uniform_broadcast(v.s01, 0);
;;     v.s012 = sub_group_non_uniform_broadcast(v.s012, 0);
;;     v.s0123 = sub_group_non_uniform_broadcast(v.s0123, 0);
;;     v.s01234567 = sub_group_non_uniform_broadcast(v.s01234567, 0);
;;     v = sub_group_non_uniform_broadcast(v, 0);
;;     v.s0 = sub_group_broadcast_first(v.s0);
;; }
;; 
;; kernel void testBallotOperations(global uint* dst)
;; {
;;     uint4 v = sub_group_ballot(0);
;;     dst[0] = sub_group_inverse_ballot(v);
;;     dst[1] = sub_group_ballot_bit_extract(v, 0);
;;     dst[2] = sub_group_ballot_bit_count(v);
;;     dst[3] = sub_group_ballot_inclusive_scan(v);
;;     dst[4] = sub_group_ballot_exclusive_scan(v);
;;     dst[5] = sub_group_ballot_find_lsb(v);
;;     dst[6] = sub_group_ballot_find_msb(v);
;; }
;; 
;; kernel void testSubgroupMasks(global uint4* dst)
;; {
;;     dst[0] = get_sub_group_eq_mask();
;;     dst[1] = get_sub_group_ge_mask();
;;     dst[2] = get_sub_group_gt_mask();
;;     dst[3] = get_sub_group_le_mask();
;;     dst[4] = get_sub_group_lt_mask();
;; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-LLVM
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefixes=CHECK-COMMON,CHECK-SPV-IR

; ModuleID = 'subgroup_ballot.cl'
source_filename = "subgroup_ballot.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV-DAG: {{[0-9]*}} Capability GroupNonUniformBallot

; CHECK-SPIRV-DAG: Decorate [[eqMask:[0-9]+]] BuiltIn 4416
; CHECK-SPIRV-DAG: Decorate [[geMask:[0-9]+]] BuiltIn 4417
; CHECK-SPIRV-DAG: Decorate [[gtMask:[0-9]+]] BuiltIn 4418
; CHECK-SPIRV-DAG: Decorate [[leMask:[0-9]+]] BuiltIn 4419
; CHECK-SPIRV-DAG: Decorate [[ltMask:[0-9]+]] BuiltIn 4420

; CHECK-SPIRV-DAG: TypeBool  [[bool:[0-9]+]]
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

; CHECK-SPIRV-DAG: ConstantFalse [[bool]] [[false:[0-9]+]]
; CHECK-SPIRV-DAG: Constant [[int]]    [[ScopeSubgroup:[0-9]+]] 3
; CHECK-SPIRV-DAG: Constant [[char]]   [[char_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[short]]  [[short_0:[0-9]+]]       0
; CHECK-SPIRV-DAG: Constant [[int]]    [[int_0:[0-9]+]]         0
; CHECK-SPIRV-DAG: Constant [[long]]   [[long_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[half]]   [[half_0:[0-9]+]]        0
; CHECK-SPIRV-DAG: Constant [[float]]  [[float_0:[0-9]+]]       0
; CHECK-SPIRV-DAG: Constant [[double]] [[double_0:[0-9]+]]      0

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char2]] [[char2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char2]] {{[0-9]+}} [[ScopeSubgroup]] [[char2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char3]] [[char3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char3]] {{[0-9]+}} [[ScopeSubgroup]] [[char3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char4]] [[char4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char4]] {{[0-9]+}} [[ScopeSubgroup]] [[char4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char8]] [[char8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char8]] {{[0-9]+}} [[ScopeSubgroup]] [[char8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[char16]] [[char16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char16]] {{[0-9]+}} [[ScopeSubgroup]] [[char16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[char]] [[char_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastChars

; CHECK-LLVM: call spir_func i8 @_Z31sub_group_non_uniform_broadcasthj(i8 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_hj(<2 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_hj(<3 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_hj(<4 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_hj(<8 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_hj(<16 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i8 @_Z25sub_group_broadcast_firsth(i8 {{.*}})

; CHECK-SPV-IR: call spir_func i8 @_Z32__spirv_GroupNonUniformBroadcasticj(i32 3, i8 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv2_cj(i32 3, <2 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv3_cj(i32 3, <3 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv4_cj(i32 3, <4 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv8_cj(i32 3, <8 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv16_cj(i32 3, <16 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i8 @_Z37__spirv_GroupNonUniformBroadcastFirstic(i32 3, i8 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastChars() local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func signext i8 @_Z31sub_group_non_uniform_broadcastcj(i8 signext 0, i32 0) #7
  %2 = insertelement <16 x i8> <i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %1, i64 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_cj(<2 x i8> %3, i32 0) #7
  %5 = shufflevector <2 x i8> %4, <2 x i8> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i8> %5, <16 x i8> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_cj(<3 x i8> %7, i32 0) #7
  %9 = shufflevector <3 x i8> %8, <3 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i8> %9, <16 x i8> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i8> %10, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_cj(<4 x i8> %11, i32 0) #7
  %13 = shufflevector <4 x i8> %12, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i8> %13, <16 x i8> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i8> %14, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_cj(<8 x i8> %15, i32 0) #7
  %17 = shufflevector <8 x i8> %16, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i8> %17, <16 x i8> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_cj(<16 x i8> %18, i32 0) #7
  %20 = extractelement <16 x i8> %19, i64 0
  %21 = tail call spir_func signext i8 @_Z25sub_group_broadcast_firstc(i8 signext %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z31sub_group_non_uniform_broadcastcj(i8 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_cj(<2 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_cj(<3 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_cj(<4 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_cj(<8 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_cj(<16 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i8 @_Z25sub_group_broadcast_firstc(i8 signext) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char2]] [[char2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char2]] {{[0-9]+}} [[ScopeSubgroup]] [[char2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char3]] [[char3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char3]] {{[0-9]+}} [[ScopeSubgroup]] [[char3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char4]] [[char4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char4]] {{[0-9]+}} [[ScopeSubgroup]] [[char4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char8]] [[char8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char8]] {{[0-9]+}} [[ScopeSubgroup]] [[char8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[char16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[char16]] [[char16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[char16]] {{[0-9]+}} [[ScopeSubgroup]] [[char16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[char]] [[char_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[char]] {{[0-9]+}} [[ScopeSubgroup]] [[char_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastUChars

; CHECK-LLVM: call spir_func i8 @_Z31sub_group_non_uniform_broadcasthj(i8 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_hj(<2 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_hj(<3 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_hj(<4 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_hj(<8 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_hj(<16 x i8> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i8 @_Z25sub_group_broadcast_firsth(i8 {{.*}})

; CHECK-SPV-IR: call spir_func i8 @_Z32__spirv_GroupNonUniformBroadcasticj(i32 3, i8 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv2_cj(i32 3, <2 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv3_cj(i32 3, <3 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv4_cj(i32 3, <4 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv8_cj(i32 3, <8 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i8> @_Z32__spirv_GroupNonUniformBroadcastiDv16_cj(i32 3, <16 x i8> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i8 @_Z37__spirv_GroupNonUniformBroadcastFirstic(i32 3, i8 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastUChars() local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func zeroext i8 @_Z31sub_group_non_uniform_broadcasthj(i8 zeroext 0, i32 0) #7
  %2 = insertelement <16 x i8> <i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %1, i64 0
  %3 = shufflevector <16 x i8> %2, <16 x i8> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_hj(<2 x i8> %3, i32 0) #7
  %5 = shufflevector <2 x i8> %4, <2 x i8> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i8> %5, <16 x i8> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i8> %6, <16 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_hj(<3 x i8> %7, i32 0) #7
  %9 = shufflevector <3 x i8> %8, <3 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i8> %9, <16 x i8> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i8> %10, <16 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_hj(<4 x i8> %11, i32 0) #7
  %13 = shufflevector <4 x i8> %12, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i8> %13, <16 x i8> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i8> %14, <16 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_hj(<8 x i8> %15, i32 0) #7
  %17 = shufflevector <8 x i8> %16, <8 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i8> %17, <16 x i8> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_hj(<16 x i8> %18, i32 0) #7
  %20 = extractelement <16 x i8> %19, i64 0
  %21 = tail call spir_func zeroext i8 @_Z25sub_group_broadcast_firsth(i8 zeroext %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z31sub_group_non_uniform_broadcasthj(i8 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i8> @_Z31sub_group_non_uniform_broadcastDv2_hj(<2 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i8> @_Z31sub_group_non_uniform_broadcastDv3_hj(<3 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i8> @_Z31sub_group_non_uniform_broadcastDv4_hj(<4 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i8> @_Z31sub_group_non_uniform_broadcastDv8_hj(<8 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i8> @_Z31sub_group_non_uniform_broadcastDv16_hj(<16 x i8>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i8 @_Z25sub_group_broadcast_firsth(i8 zeroext) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short2]] [[short2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short2]] {{[0-9]+}} [[ScopeSubgroup]] [[short2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short3]] [[short3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short3]] {{[0-9]+}} [[ScopeSubgroup]] [[short3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short4]] [[short4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short4]] {{[0-9]+}} [[ScopeSubgroup]] [[short4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short8]] [[short8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short8]] {{[0-9]+}} [[ScopeSubgroup]] [[short8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[short16]] [[short16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short16]] {{[0-9]+}} [[ScopeSubgroup]] [[short16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[short]] [[short_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastShorts

; CHECK-LLVM: call spir_func i16 @_Z31sub_group_non_uniform_broadcasttj(i16 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_tj(<2 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_tj(<3 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_tj(<4 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_tj(<8 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_tj(<16 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i16 @_Z25sub_group_broadcast_firstt(i16 {{.*}})

; CHECK-SPV-IR: call spir_func i16 @_Z32__spirv_GroupNonUniformBroadcastisj(i32 3, i16 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv2_sj(i32 3, <2 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv3_sj(i32 3, <3 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv4_sj(i32 3, <4 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv8_sj(i32 3, <8 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv16_sj(i32 3, <16 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i16 @_Z37__spirv_GroupNonUniformBroadcastFirstis(i32 3, i16 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastShorts() local_unnamed_addr #2 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func signext i16 @_Z31sub_group_non_uniform_broadcastsj(i16 signext 0, i32 0) #7
  %2 = insertelement <16 x i16> <i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %1, i64 0
  %3 = shufflevector <16 x i16> %2, <16 x i16> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_sj(<2 x i16> %3, i32 0) #7
  %5 = shufflevector <2 x i16> %4, <2 x i16> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i16> %5, <16 x i16> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i16> %6, <16 x i16> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_sj(<3 x i16> %7, i32 0) #7
  %9 = shufflevector <3 x i16> %8, <3 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i16> %9, <16 x i16> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i16> %10, <16 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_sj(<4 x i16> %11, i32 0) #7
  %13 = shufflevector <4 x i16> %12, <4 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i16> %13, <16 x i16> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i16> %14, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_sj(<8 x i16> %15, i32 0) #7
  %17 = shufflevector <8 x i16> %16, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i16> %17, <16 x i16> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_sj(<16 x i16> %18, i32 0) #7
  %20 = extractelement <16 x i16> %19, i64 0
  %21 = tail call spir_func signext i16 @_Z25sub_group_broadcast_firsts(i16 signext %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z31sub_group_non_uniform_broadcastsj(i16 signext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_sj(<2 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_sj(<3 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_sj(<4 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_sj(<8 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_sj(<16 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func signext i16 @_Z25sub_group_broadcast_firsts(i16 signext) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short2]] [[short2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short2]] {{[0-9]+}} [[ScopeSubgroup]] [[short2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short3]] [[short3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short3]] {{[0-9]+}} [[ScopeSubgroup]] [[short3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short4]] [[short4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short4]] {{[0-9]+}} [[ScopeSubgroup]] [[short4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short8]] [[short8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short8]] {{[0-9]+}} [[ScopeSubgroup]] [[short8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[short16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[short16]] [[short16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[short16]] {{[0-9]+}} [[ScopeSubgroup]] [[short16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[short]] [[short_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[short]] {{[0-9]+}} [[ScopeSubgroup]] [[short_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastUShorts

; CHECK-LLVM: call spir_func i16 @_Z31sub_group_non_uniform_broadcasttj(i16 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_tj(<2 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_tj(<3 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_tj(<4 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_tj(<8 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_tj(<16 x i16> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i16 @_Z25sub_group_broadcast_firstt(i16 {{.*}})

; CHECK-SPV-IR: call spir_func i16 @_Z32__spirv_GroupNonUniformBroadcastisj(i32 3, i16 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv2_sj(i32 3, <2 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv3_sj(i32 3, <3 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv4_sj(i32 3, <4 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv8_sj(i32 3, <8 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i16> @_Z32__spirv_GroupNonUniformBroadcastiDv16_sj(i32 3, <16 x i16> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i16 @_Z37__spirv_GroupNonUniformBroadcastFirstis(i32 3, i16 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastUShorts() local_unnamed_addr #2 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func zeroext i16 @_Z31sub_group_non_uniform_broadcasttj(i16 zeroext 0, i32 0) #7
  %2 = insertelement <16 x i16> <i16 undef, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %1, i64 0
  %3 = shufflevector <16 x i16> %2, <16 x i16> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_tj(<2 x i16> %3, i32 0) #7
  %5 = shufflevector <2 x i16> %4, <2 x i16> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i16> %5, <16 x i16> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i16> %6, <16 x i16> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_tj(<3 x i16> %7, i32 0) #7
  %9 = shufflevector <3 x i16> %8, <3 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i16> %9, <16 x i16> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i16> %10, <16 x i16> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_tj(<4 x i16> %11, i32 0) #7
  %13 = shufflevector <4 x i16> %12, <4 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i16> %13, <16 x i16> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i16> %14, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_tj(<8 x i16> %15, i32 0) #7
  %17 = shufflevector <8 x i16> %16, <8 x i16> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i16> %17, <16 x i16> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_tj(<16 x i16> %18, i32 0) #7
  %20 = extractelement <16 x i16> %19, i64 0
  %21 = tail call spir_func zeroext i16 @_Z25sub_group_broadcast_firstt(i16 zeroext %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z31sub_group_non_uniform_broadcasttj(i16 zeroext, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i16> @_Z31sub_group_non_uniform_broadcastDv2_tj(<2 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i16> @_Z31sub_group_non_uniform_broadcastDv3_tj(<3 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i16> @_Z31sub_group_non_uniform_broadcastDv4_tj(<4 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i16> @_Z31sub_group_non_uniform_broadcastDv8_tj(<8 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i16> @_Z31sub_group_non_uniform_broadcastDv16_tj(<16 x i16>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func zeroext i16 @_Z25sub_group_broadcast_firstt(i16 zeroext) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int2]] [[int2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int2]] {{[0-9]+}} [[ScopeSubgroup]] [[int2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int3]] [[int3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int3]] {{[0-9]+}} [[ScopeSubgroup]] [[int3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int4]] [[int4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int4]] {{[0-9]+}} [[ScopeSubgroup]] [[int4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int8]] [[int8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int8]] {{[0-9]+}} [[ScopeSubgroup]] [[int8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[int16]] [[int16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int16]] {{[0-9]+}} [[ScopeSubgroup]] [[int16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[int]] [[int_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastInts

; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_broadcastjj(i32 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_jj(<2 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_jj(<3 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_jj(<4 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_jj(<8 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_jj(<16 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i32 @_Z25sub_group_broadcast_firstj(i32 {{.*}})

; CHECK-SPV-IR: call spir_func i32 @_Z32__spirv_GroupNonUniformBroadcastiij(i32 3, i32 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv2_ij(i32 3, <2 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv3_ij(i32 3, <3 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv4_ij(i32 3, <4 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv8_ij(i32 3, <8 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv16_ij(i32 3, <16 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i32 @_Z37__spirv_GroupNonUniformBroadcastFirstii(i32 3, i32 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastInts() local_unnamed_addr #3 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i32 @_Z31sub_group_non_uniform_broadcastij(i32 0, i32 0) #7
  %2 = insertelement <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %1, i64 0
  %3 = shufflevector <16 x i32> %2, <16 x i32> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_ij(<2 x i32> %3, i32 0) #7
  %5 = shufflevector <2 x i32> %4, <2 x i32> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i32> %5, <16 x i32> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i32> %6, <16 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_ij(<3 x i32> %7, i32 0) #7
  %9 = shufflevector <3 x i32> %8, <3 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i32> %9, <16 x i32> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i32> %10, <16 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_ij(<4 x i32> %11, i32 0) #7
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i32> %13, <16 x i32> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i32> %14, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_ij(<8 x i32> %15, i32 0) #7
  %17 = shufflevector <8 x i32> %16, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i32> %17, <16 x i32> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_ij(<16 x i32> %18, i32 0) #7
  %20 = extractelement <16 x i32> %19, i64 0
  %21 = tail call spir_func i32 @_Z25sub_group_broadcast_firsti(i32 %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_broadcastij(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_ij(<2 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_ij(<3 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_ij(<4 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_ij(<8 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_ij(<16 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25sub_group_broadcast_firsti(i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int2]] [[int2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int2]] {{[0-9]+}} [[ScopeSubgroup]] [[int2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int3]] [[int3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int3]] {{[0-9]+}} [[ScopeSubgroup]] [[int3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int4]] [[int4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int4]] {{[0-9]+}} [[ScopeSubgroup]] [[int4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int8]] [[int8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int8]] {{[0-9]+}} [[ScopeSubgroup]] [[int8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[int16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[int16]] [[int16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[int16]] {{[0-9]+}} [[ScopeSubgroup]] [[int16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[int]] [[int_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[int_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastUInts

; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_broadcastjj(i32 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_jj(<2 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_jj(<3 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_jj(<4 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_jj(<8 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_jj(<16 x i32> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i32 @_Z25sub_group_broadcast_firstj(i32 {{.*}})

; CHECK-SPV-IR: call spir_func i32 @_Z32__spirv_GroupNonUniformBroadcastiij(i32 3, i32 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv2_ij(i32 3, <2 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv3_ij(i32 3, <3 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv4_ij(i32 3, <4 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv8_ij(i32 3, <8 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i32> @_Z32__spirv_GroupNonUniformBroadcastiDv16_ij(i32 3, <16 x i32> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i32 @_Z37__spirv_GroupNonUniformBroadcastFirstii(i32 3, i32 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastUInts() local_unnamed_addr #3 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i32 @_Z31sub_group_non_uniform_broadcastjj(i32 0, i32 0) #7
  %2 = insertelement <16 x i32> <i32 undef, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %1, i64 0
  %3 = shufflevector <16 x i32> %2, <16 x i32> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_jj(<2 x i32> %3, i32 0) #7
  %5 = shufflevector <2 x i32> %4, <2 x i32> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i32> %5, <16 x i32> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i32> %6, <16 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_jj(<3 x i32> %7, i32 0) #7
  %9 = shufflevector <3 x i32> %8, <3 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i32> %9, <16 x i32> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i32> %10, <16 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_jj(<4 x i32> %11, i32 0) #7
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i32> %13, <16 x i32> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i32> %14, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_jj(<8 x i32> %15, i32 0) #7
  %17 = shufflevector <8 x i32> %16, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i32> %17, <16 x i32> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_jj(<16 x i32> %18, i32 0) #7
  %20 = extractelement <16 x i32> %19, i64 0
  %21 = tail call spir_func i32 @_Z25sub_group_broadcast_firstj(i32 %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_broadcastjj(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i32> @_Z31sub_group_non_uniform_broadcastDv2_jj(<2 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i32> @_Z31sub_group_non_uniform_broadcastDv3_jj(<3 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i32> @_Z31sub_group_non_uniform_broadcastDv4_jj(<4 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i32> @_Z31sub_group_non_uniform_broadcastDv8_jj(<8 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i32> @_Z31sub_group_non_uniform_broadcastDv16_jj(<16 x i32>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25sub_group_broadcast_firstj(i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long2]] [[long2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long2]] {{[0-9]+}} [[ScopeSubgroup]] [[long2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long3]] [[long3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long3]] {{[0-9]+}} [[ScopeSubgroup]] [[long3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long4]] [[long4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long4]] {{[0-9]+}} [[ScopeSubgroup]] [[long4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long8]] [[long8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long8]] {{[0-9]+}} [[ScopeSubgroup]] [[long8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[long16]] [[long16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long16]] {{[0-9]+}} [[ScopeSubgroup]] [[long16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[long]] [[long_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastLongs

; CHECK-LLVM: call spir_func i64 @_Z31sub_group_non_uniform_broadcastmj(i64 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_mj(<2 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_mj(<3 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_mj(<4 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_mj(<8 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_mj(<16 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i64 @_Z25sub_group_broadcast_firstm(i64 {{.*}})

; CHECK-SPV-IR: call spir_func i64 @_Z32__spirv_GroupNonUniformBroadcastilj(i32 3, i64 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv2_lj(i32 3, <2 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv3_lj(i32 3, <3 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv4_lj(i32 3, <4 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv8_lj(i32 3, <8 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv16_lj(i32 3, <16 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i64 @_Z37__spirv_GroupNonUniformBroadcastFirstil(i32 3, i64 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastLongs() local_unnamed_addr #4 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i64 @_Z31sub_group_non_uniform_broadcastlj(i64 0, i32 0) #7
  %2 = insertelement <16 x i64> <i64 undef, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0>, i64 %1, i64 0
  %3 = shufflevector <16 x i64> %2, <16 x i64> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_lj(<2 x i64> %3, i32 0) #7
  %5 = shufflevector <2 x i64> %4, <2 x i64> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i64> %5, <16 x i64> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i64> %6, <16 x i64> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_lj(<3 x i64> %7, i32 0) #7
  %9 = shufflevector <3 x i64> %8, <3 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i64> %9, <16 x i64> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i64> %10, <16 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_lj(<4 x i64> %11, i32 0) #7
  %13 = shufflevector <4 x i64> %12, <4 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i64> %13, <16 x i64> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i64> %14, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_lj(<8 x i64> %15, i32 0) #7
  %17 = shufflevector <8 x i64> %16, <8 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i64> %17, <16 x i64> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_lj(<16 x i64> %18, i32 0) #7
  %20 = extractelement <16 x i64> %19, i64 0
  %21 = tail call spir_func i64 @_Z25sub_group_broadcast_firstl(i64 %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z31sub_group_non_uniform_broadcastlj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_lj(<2 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_lj(<3 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_lj(<4 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_lj(<8 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_lj(<16 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z25sub_group_broadcast_firstl(i64) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long2]] [[long2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long2]] {{[0-9]+}} [[ScopeSubgroup]] [[long2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long3]] [[long3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long3]] {{[0-9]+}} [[ScopeSubgroup]] [[long3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long4]] [[long4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long4]] {{[0-9]+}} [[ScopeSubgroup]] [[long4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long8]] [[long8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long8]] {{[0-9]+}} [[ScopeSubgroup]] [[long8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[long16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[long16]] [[long16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[long16]] {{[0-9]+}} [[ScopeSubgroup]] [[long16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[long]] [[long_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[long]] {{[0-9]+}} [[ScopeSubgroup]] [[long_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastULongs

; CHECK-LLVM: call spir_func i64 @_Z31sub_group_non_uniform_broadcastmj(i64 {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_mj(<2 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_mj(<3 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_mj(<4 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_mj(<8 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_mj(<16 x i64> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func i64 @_Z25sub_group_broadcast_firstm(i64 {{.*}})

; CHECK-SPV-IR: call spir_func i64 @_Z32__spirv_GroupNonUniformBroadcastilj(i32 3, i64 {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv2_lj(i32 3, <2 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv3_lj(i32 3, <3 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv4_lj(i32 3, <4 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv8_lj(i32 3, <8 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x i64> @_Z32__spirv_GroupNonUniformBroadcastiDv16_lj(i32 3, <16 x i64> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func i64 @_Z37__spirv_GroupNonUniformBroadcastFirstil(i32 3, i64 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastULongs() local_unnamed_addr #4 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func i64 @_Z31sub_group_non_uniform_broadcastmj(i64 0, i32 0) #7
  %2 = insertelement <16 x i64> <i64 undef, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0>, i64 %1, i64 0
  %3 = shufflevector <16 x i64> %2, <16 x i64> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_mj(<2 x i64> %3, i32 0) #7
  %5 = shufflevector <2 x i64> %4, <2 x i64> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x i64> %5, <16 x i64> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x i64> %6, <16 x i64> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_mj(<3 x i64> %7, i32 0) #7
  %9 = shufflevector <3 x i64> %8, <3 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x i64> %9, <16 x i64> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x i64> %10, <16 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_mj(<4 x i64> %11, i32 0) #7
  %13 = shufflevector <4 x i64> %12, <4 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x i64> %13, <16 x i64> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x i64> %14, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_mj(<8 x i64> %15, i32 0) #7
  %17 = shufflevector <8 x i64> %16, <8 x i64> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x i64> %17, <16 x i64> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_mj(<16 x i64> %18, i32 0) #7
  %20 = extractelement <16 x i64> %19, i64 0
  %21 = tail call spir_func i64 @_Z25sub_group_broadcast_firstm(i64 %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z31sub_group_non_uniform_broadcastmj(i64, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x i64> @_Z31sub_group_non_uniform_broadcastDv2_mj(<2 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x i64> @_Z31sub_group_non_uniform_broadcastDv3_mj(<3 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x i64> @_Z31sub_group_non_uniform_broadcastDv4_mj(<4 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x i64> @_Z31sub_group_non_uniform_broadcastDv8_mj(<8 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x i64> @_Z31sub_group_non_uniform_broadcastDv16_mj(<16 x i64>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z25sub_group_broadcast_firstm(i64) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float2]] [[float2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[float2]] {{[0-9]+}} [[ScopeSubgroup]] [[float2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float3]] [[float3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[float3]] {{[0-9]+}} [[ScopeSubgroup]] [[float3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float4]] [[float4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[float4]] {{[0-9]+}} [[ScopeSubgroup]] [[float4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float8]] [[float8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[float8]] {{[0-9]+}} [[ScopeSubgroup]] [[float8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[float16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[float16]] [[float16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[float16]] {{[0-9]+}} [[ScopeSubgroup]] [[float16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[float]] [[float_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[float]] {{[0-9]+}} [[ScopeSubgroup]] [[float_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastFloats

; CHECK-LLVM: call spir_func float @_Z31sub_group_non_uniform_broadcastfj(float {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x float> @_Z31sub_group_non_uniform_broadcastDv2_fj(<2 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x float> @_Z31sub_group_non_uniform_broadcastDv3_fj(<3 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x float> @_Z31sub_group_non_uniform_broadcastDv4_fj(<4 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x float> @_Z31sub_group_non_uniform_broadcastDv8_fj(<8 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x float> @_Z31sub_group_non_uniform_broadcastDv16_fj(<16 x float> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func float @_Z25sub_group_broadcast_firstf(float {{.*}})

; CHECK-SPV-IR: call spir_func float @_Z32__spirv_GroupNonUniformBroadcastifj(i32 3, float {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x float> @_Z32__spirv_GroupNonUniformBroadcastiDv2_fj(i32 3, <2 x float> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x float> @_Z32__spirv_GroupNonUniformBroadcastiDv3_fj(i32 3, <3 x float> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x float> @_Z32__spirv_GroupNonUniformBroadcastiDv4_fj(i32 3, <4 x float> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x float> @_Z32__spirv_GroupNonUniformBroadcastiDv8_fj(i32 3, <8 x float> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x float> @_Z32__spirv_GroupNonUniformBroadcastiDv16_fj(i32 3, <16 x float> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func float @_Z37__spirv_GroupNonUniformBroadcastFirstif(i32 3, float {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastFloats() local_unnamed_addr #3 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func float @_Z31sub_group_non_uniform_broadcastfj(float 0.000000e+00, i32 0) #7
  %2 = insertelement <16 x float> <float undef, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %1, i64 0
  %3 = shufflevector <16 x float> %2, <16 x float> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x float> @_Z31sub_group_non_uniform_broadcastDv2_fj(<2 x float> %3, i32 0) #7
  %5 = shufflevector <2 x float> %4, <2 x float> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x float> %5, <16 x float> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x float> %6, <16 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x float> @_Z31sub_group_non_uniform_broadcastDv3_fj(<3 x float> %7, i32 0) #7
  %9 = shufflevector <3 x float> %8, <3 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x float> %9, <16 x float> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x float> %10, <16 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x float> @_Z31sub_group_non_uniform_broadcastDv4_fj(<4 x float> %11, i32 0) #7
  %13 = shufflevector <4 x float> %12, <4 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x float> %13, <16 x float> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x float> %14, <16 x float> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x float> @_Z31sub_group_non_uniform_broadcastDv8_fj(<8 x float> %15, i32 0) #7
  %17 = shufflevector <8 x float> %16, <8 x float> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x float> %17, <16 x float> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x float> @_Z31sub_group_non_uniform_broadcastDv16_fj(<16 x float> %18, i32 0) #7
  %20 = extractelement <16 x float> %19, i64 0
  %21 = tail call spir_func float @_Z25sub_group_broadcast_firstf(float %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func float @_Z31sub_group_non_uniform_broadcastfj(float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x float> @_Z31sub_group_non_uniform_broadcastDv2_fj(<2 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x float> @_Z31sub_group_non_uniform_broadcastDv3_fj(<3 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x float> @_Z31sub_group_non_uniform_broadcastDv4_fj(<4 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x float> @_Z31sub_group_non_uniform_broadcastDv8_fj(<8 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x float> @_Z31sub_group_non_uniform_broadcastDv16_fj(<16 x float>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func float @_Z25sub_group_broadcast_firstf(float) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half2]] [[half2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[half2]] {{[0-9]+}} [[ScopeSubgroup]] [[half2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half3]] [[half3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[half3]] {{[0-9]+}} [[ScopeSubgroup]] [[half3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half4]] [[half4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[half4]] {{[0-9]+}} [[ScopeSubgroup]] [[half4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half8]] [[half8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[half8]] {{[0-9]+}} [[ScopeSubgroup]] [[half8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[half16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[half16]] [[half16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[half16]] {{[0-9]+}} [[ScopeSubgroup]] [[half16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[half]] [[half_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[half]] {{[0-9]+}} [[ScopeSubgroup]] [[half_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastHalfs

; CHECK-LLVM: call spir_func half @_Z31sub_group_non_uniform_broadcastDhj(half {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x half> @_Z31sub_group_non_uniform_broadcastDv2_Dhj(<2 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x half> @_Z31sub_group_non_uniform_broadcastDv3_Dhj(<3 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x half> @_Z31sub_group_non_uniform_broadcastDv4_Dhj(<4 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x half> @_Z31sub_group_non_uniform_broadcastDv8_Dhj(<8 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x half> @_Z31sub_group_non_uniform_broadcastDv16_Dhj(<16 x half> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func half @_Z25sub_group_broadcast_firstDh(half {{.*}})

; CHECK-SPV-IR: call spir_func half @_Z32__spirv_GroupNonUniformBroadcastiDhj(i32 3, half {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x half> @_Z32__spirv_GroupNonUniformBroadcastiDv2_Dhj(i32 3, <2 x half> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x half> @_Z32__spirv_GroupNonUniformBroadcastiDv3_Dhj(i32 3, <3 x half> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x half> @_Z32__spirv_GroupNonUniformBroadcastiDv4_Dhj(i32 3, <4 x half> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x half> @_Z32__spirv_GroupNonUniformBroadcastiDv8_Dhj(i32 3, <8 x half> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x half> @_Z32__spirv_GroupNonUniformBroadcastiDv16_Dhj(i32 3, <16 x half> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func half @_Z37__spirv_GroupNonUniformBroadcastFirstiDh(i32 3, half {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastHalfs() local_unnamed_addr #2 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func half @_Z31sub_group_non_uniform_broadcastDhj(half 0xH0000, i32 0) #7
  %2 = insertelement <16 x half> <half undef, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000, half 0xH0000>, half %1, i64 0
  %3 = shufflevector <16 x half> %2, <16 x half> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x half> @_Z31sub_group_non_uniform_broadcastDv2_Dhj(<2 x half> %3, i32 0) #7
  %5 = shufflevector <2 x half> %4, <2 x half> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x half> %5, <16 x half> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x half> %6, <16 x half> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x half> @_Z31sub_group_non_uniform_broadcastDv3_Dhj(<3 x half> %7, i32 0) #7
  %9 = shufflevector <3 x half> %8, <3 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x half> %9, <16 x half> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x half> %10, <16 x half> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x half> @_Z31sub_group_non_uniform_broadcastDv4_Dhj(<4 x half> %11, i32 0) #7
  %13 = shufflevector <4 x half> %12, <4 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x half> %13, <16 x half> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x half> %14, <16 x half> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x half> @_Z31sub_group_non_uniform_broadcastDv8_Dhj(<8 x half> %15, i32 0) #7
  %17 = shufflevector <8 x half> %16, <8 x half> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x half> %17, <16 x half> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x half> @_Z31sub_group_non_uniform_broadcastDv16_Dhj(<16 x half> %18, i32 0) #7
  %20 = extractelement <16 x half> %19, i64 0
  %21 = tail call spir_func half @_Z25sub_group_broadcast_firstDh(half %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func half @_Z31sub_group_non_uniform_broadcastDhj(half, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x half> @_Z31sub_group_non_uniform_broadcastDv2_Dhj(<2 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x half> @_Z31sub_group_non_uniform_broadcastDv3_Dhj(<3 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x half> @_Z31sub_group_non_uniform_broadcastDv4_Dhj(<4 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x half> @_Z31sub_group_non_uniform_broadcastDv8_Dhj(<8 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x half> @_Z31sub_group_non_uniform_broadcastDv16_Dhj(<16 x half>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func half @_Z25sub_group_broadcast_firstDh(half) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBroadcast [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double2]] [[double2_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[double2]] {{[0-9]+}} [[ScopeSubgroup]] [[double2_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double3]] [[double3_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[double3]] {{[0-9]+}} [[ScopeSubgroup]] [[double3_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double4]] [[double4_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[double4]] {{[0-9]+}} [[ScopeSubgroup]] [[double4_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double8]] [[double8_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[double8]] {{[0-9]+}} [[ScopeSubgroup]] [[double8_0]] [[int_0]]
; CHECK-SPIRV: VectorShuffle [[double16]] {{[0-9]+}}
; CHECK-SPIRV: VectorShuffle [[double16]] [[double16_0:[0-9]+]] 
; CHECK-SPIRV: GroupNonUniformBroadcast [[double16]] {{[0-9]+}} [[ScopeSubgroup]] [[double16_0]] [[int_0]]
; CHECK-SPIRV: CompositeExtract [[double]] [[double_value:[0-9]+]]
; CHECK-SPIRV: GroupNonUniformBroadcastFirst [[double]] {{[0-9]+}} [[ScopeSubgroup]] [[double_value]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testNonUniformBroadcastDoubles

; CHECK-LLVM: call spir_func double @_Z31sub_group_non_uniform_broadcastdj(double {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <2 x double> @_Z31sub_group_non_uniform_broadcastDv2_dj(<2 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <3 x double> @_Z31sub_group_non_uniform_broadcastDv3_dj(<3 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <4 x double> @_Z31sub_group_non_uniform_broadcastDv4_dj(<4 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <8 x double> @_Z31sub_group_non_uniform_broadcastDv8_dj(<8 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func <16 x double> @_Z31sub_group_non_uniform_broadcastDv16_dj(<16 x double> {{.*}}, i32 0)
; CHECK-LLVM: call spir_func double @_Z25sub_group_broadcast_firstd(double {{.*}})

; CHECK-SPV-IR: call spir_func double @_Z32__spirv_GroupNonUniformBroadcastidj(i32 3, double {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <2 x double> @_Z32__spirv_GroupNonUniformBroadcastiDv2_dj(i32 3, <2 x double> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <3 x double> @_Z32__spirv_GroupNonUniformBroadcastiDv3_dj(i32 3, <3 x double> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <4 x double> @_Z32__spirv_GroupNonUniformBroadcastiDv4_dj(i32 3, <4 x double> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <8 x double> @_Z32__spirv_GroupNonUniformBroadcastiDv8_dj(i32 3, <8 x double> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func <16 x double> @_Z32__spirv_GroupNonUniformBroadcastiDv16_dj(i32 3, <16 x double> {{.*}}, i32 0)
; CHECK-SPV-IR: call spir_func double @_Z37__spirv_GroupNonUniformBroadcastFirstid(i32 3, double {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testNonUniformBroadcastDoubles() local_unnamed_addr #4 !kernel_arg_addr_space !3 !kernel_arg_access_qual !3 !kernel_arg_type !3 !kernel_arg_base_type !3 !kernel_arg_type_qual !3 {
  %1 = tail call spir_func double @_Z31sub_group_non_uniform_broadcastdj(double 0.000000e+00, i32 0) #7
  %2 = insertelement <16 x double> <double undef, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00>, double %1, i64 0
  %3 = shufflevector <16 x double> %2, <16 x double> undef, <2 x i32> <i32 0, i32 1>
  %4 = tail call spir_func <2 x double> @_Z31sub_group_non_uniform_broadcastDv2_dj(<2 x double> %3, i32 0) #7
  %5 = shufflevector <2 x double> %4, <2 x double> undef, <16 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %6 = shufflevector <16 x double> %5, <16 x double> %2, <16 x i32> <i32 0, i32 1, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %7 = shufflevector <16 x double> %6, <16 x double> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %8 = tail call spir_func <3 x double> @_Z31sub_group_non_uniform_broadcastDv3_dj(<3 x double> %7, i32 0) #7
  %9 = shufflevector <3 x double> %8, <3 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %10 = shufflevector <16 x double> %9, <16 x double> %6, <16 x i32> <i32 0, i32 1, i32 2, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = shufflevector <16 x double> %10, <16 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %12 = tail call spir_func <4 x double> @_Z31sub_group_non_uniform_broadcastDv4_dj(<4 x double> %11, i32 0) #7
  %13 = shufflevector <4 x double> %12, <4 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %14 = shufflevector <16 x double> %13, <16 x double> %10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %15 = shufflevector <16 x double> %14, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %16 = tail call spir_func <8 x double> @_Z31sub_group_non_uniform_broadcastDv8_dj(<8 x double> %15, i32 0) #7
  %17 = shufflevector <8 x double> %16, <8 x double> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %18 = shufflevector <16 x double> %17, <16 x double> %14, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %19 = tail call spir_func <16 x double> @_Z31sub_group_non_uniform_broadcastDv16_dj(<16 x double> %18, i32 0) #7
  %20 = extractelement <16 x double> %19, i64 0
  %21 = tail call spir_func double @_Z25sub_group_broadcast_firstd(double %20) #7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func double @_Z31sub_group_non_uniform_broadcastdj(double, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <2 x double> @_Z31sub_group_non_uniform_broadcastDv2_dj(<2 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <3 x double> @_Z31sub_group_non_uniform_broadcastDv3_dj(<3 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <4 x double> @_Z31sub_group_non_uniform_broadcastDv4_dj(<4 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <8 x double> @_Z31sub_group_non_uniform_broadcastDv8_dj(<8 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func <16 x double> @_Z31sub_group_non_uniform_broadcastDv16_dj(<16 x double>, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func double @_Z25sub_group_broadcast_firstd(double) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformBallot [[int4]] [[ballot:[0-9]+]] [[ScopeSubgroup]] [[false]]
; CHECK-SPIRV: GroupNonUniformInverseBallot [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[ballot]]
; CHECK-SPIRV: GroupNonUniformBallotBitExtract [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[ballot]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformBallotBitCount [[int]] {{[0-9]+}} [[ScopeSubgroup]] 0 [[ballot]]
; CHECK-SPIRV: GroupNonUniformBallotBitCount [[int]] {{[0-9]+}} [[ScopeSubgroup]] 1 [[ballot]]
; CHECK-SPIRV: GroupNonUniformBallotBitCount [[int]] {{[0-9]+}} [[ScopeSubgroup]] 2 [[ballot]]
; CHECK-SPIRV: GroupNonUniformBallotFindLSB [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[ballot]]
; CHECK-SPIRV: GroupNonUniformBallotFindMSB [[int]] {{[0-9]+}} [[ScopeSubgroup]] [[ballot]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testBallotOperations

; CHECK-LLVM: %[[ballot:[0-9]+]] = call spir_func <4 x i32> @_Z16sub_group_balloti(i32 {{.*}})
; CHECK-LLVM: %[[inverse_ballot:[0-9]+]] = call spir_func i32 @_Z24sub_group_inverse_ballotDv4_j(<4 x i32> %[[ballot]])
; CHECK-LLVM-NEXT: icmp ne i32 %[[inverse_ballot]], 0
; CHECK-LLVM: %[[bit_extract:[0-9]+]] = call spir_func i32 @_Z28sub_group_ballot_bit_extractDv4_jj(<4 x i32> %[[ballot]], i32 0)
; CHECK-LLVM-NEXT: icmp ne i32 %[[bit_extract]], 0
; CHECK-LLVM: call spir_func i32 @_Z26sub_group_ballot_bit_countDv4_j(<4 x i32> %[[ballot]])
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_ballot_inclusive_scanDv4_j(<4 x i32> %[[ballot]])
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_ballot_exclusive_scanDv4_j(<4 x i32> %[[ballot]])
; CHECK-LLVM: call spir_func i32 @_Z25sub_group_ballot_find_lsbDv4_j(<4 x i32> %[[ballot]])
; CHECK-LLVM: call spir_func i32 @_Z25sub_group_ballot_find_msbDv4_j(<4 x i32> %[[ballot]])

; CHECK-SPV-IR: %[[ballot:[0-9]+]] = call spir_func <4 x i32> @_Z29__spirv_GroupNonUniformBallotib(i32 3, i1 false)
; CHECK-SPV-IR: call spir_func i1 @_Z36__spirv_GroupNonUniformInverseBallotiDv4_j(i32 3, <4 x i32> %[[ballot]])
; CHECK-SPV-IR: call spir_func i1 @_Z39__spirv_GroupNonUniformBallotBitExtractiDv4_jj(i32 3, <4 x i32> %[[ballot]], i32 0)
; CHECK-SPV-IR: call spir_func i32 @_Z37__spirv_GroupNonUniformBallotBitCountiiDv4_j(i32 3, i32 0, <4 x i32> %[[ballot]])
; CHECK-SPV-IR: call spir_func i32 @_Z37__spirv_GroupNonUniformBallotBitCountiiDv4_j(i32 3, i32 1, <4 x i32> %[[ballot]])
; CHECK-SPV-IR: call spir_func i32 @_Z37__spirv_GroupNonUniformBallotBitCountiiDv4_j(i32 3, i32 2, <4 x i32> %[[ballot]])
; CHECK-SPV-IR: call spir_func i32 @_Z36__spirv_GroupNonUniformBallotFindLSBiDv4_j(i32 3, <4 x i32> %[[ballot]])
; CHECK-SPV-IR: call spir_func i32 @_Z36__spirv_GroupNonUniformBallotFindMSBiDv4_j(i32 3, <4 x i32> %[[ballot]])

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testBallotOperations(i32 addrspace(1)* nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  %2 = tail call spir_func <4 x i32> @_Z16sub_group_balloti(i32 0) #7
  %3 = tail call spir_func i32 @_Z24sub_group_inverse_ballotDv4_j(<4 x i32> %2) #8
  store i32 %3, i32 addrspace(1)* %0, align 4, !tbaa !8
  %4 = tail call spir_func i32 @_Z28sub_group_ballot_bit_extractDv4_jj(<4 x i32> %2, i32 0) #8
  %5 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  store i32 %4, i32 addrspace(1)* %5, align 4, !tbaa !8
  %6 = tail call spir_func i32 @_Z26sub_group_ballot_bit_countDv4_j(<4 x i32> %2) #8
  %7 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 2
  store i32 %6, i32 addrspace(1)* %7, align 4, !tbaa !8
  %8 = tail call spir_func i32 @_Z31sub_group_ballot_inclusive_scanDv4_j(<4 x i32> %2) #7
  %9 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 3
  store i32 %8, i32 addrspace(1)* %9, align 4, !tbaa !8
  %10 = tail call spir_func i32 @_Z31sub_group_ballot_exclusive_scanDv4_j(<4 x i32> %2) #7
  %11 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 4
  store i32 %10, i32 addrspace(1)* %11, align 4, !tbaa !8
  %12 = tail call spir_func i32 @_Z25sub_group_ballot_find_lsbDv4_j(<4 x i32> %2) #7
  %13 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 5
  store i32 %12, i32 addrspace(1)* %13, align 4, !tbaa !8
  %14 = tail call spir_func i32 @_Z25sub_group_ballot_find_msbDv4_j(<4 x i32> %2) #7
  %15 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 6
  store i32 %14, i32 addrspace(1)* %15, align 4, !tbaa !8
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func <4 x i32> @_Z16sub_group_balloti(i32) local_unnamed_addr #1

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z24sub_group_inverse_ballotDv4_j(<4 x i32>) local_unnamed_addr #5

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z28sub_group_ballot_bit_extractDv4_jj(<4 x i32>, i32) local_unnamed_addr #5

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func i32 @_Z26sub_group_ballot_bit_countDv4_j(<4 x i32>) local_unnamed_addr #5

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_ballot_inclusive_scanDv4_j(<4 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_ballot_exclusive_scanDv4_j(<4 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25sub_group_ballot_find_lsbDv4_j(<4 x i32>) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25sub_group_ballot_find_msbDv4_j(<4 x i32>) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: Load [[int4]] {{[0-9]+}} [[eqMask]]
; CHECK-SPIRV: Load [[int4]] {{[0-9]+}} [[geMask]]
; CHECK-SPIRV: Load [[int4]] {{[0-9]+}} [[gtMask]]
; CHECK-SPIRV: Load [[int4]] {{[0-9]+}} [[leMask]]
; CHECK-SPIRV: Load [[int4]] {{[0-9]+}} [[ltMask]]
; CHECK-SPIRV: FunctionEnd

; CHECK-COMMON-LABEL: @testSubgroupMasks

; CHECK-LLVM: call spir_func <4 x i32> @_Z21get_sub_group_eq_maskv()
; CHECK-LLVM: call spir_func <4 x i32> @_Z21get_sub_group_ge_maskv()
; CHECK-LLVM: call spir_func <4 x i32> @_Z21get_sub_group_gt_maskv()
; CHECK-LLVM: call spir_func <4 x i32> @_Z21get_sub_group_le_maskv()
; CHECK-LLVM: call spir_func <4 x i32> @_Z21get_sub_group_lt_maskv()

; CHECK-SPV-IR: call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupEqMaskKHRv()
; CHECK-SPV-IR: call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupGeMaskKHRv()
; CHECK-SPV-IR: call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupGtMaskKHRv()
; CHECK-SPV-IR: call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupLeMaskKHRv()
; CHECK-SPV-IR: call spir_func <4 x i32> @_Z32__spirv_BuiltInSubgroupLtMaskKHRv()

; Function Attrs: convergent nofree nounwind writeonly
define dso_local spir_kernel void @testSubgroupMasks(<4 x i32> addrspace(1)* nocapture) local_unnamed_addr #6 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !12 !kernel_arg_base_type !13 !kernel_arg_type_qual !7 {
  %2 = tail call spir_func <4 x i32> @_Z21get_sub_group_eq_maskv() #8
  store <4 x i32> %2, <4 x i32> addrspace(1)* %0, align 16, !tbaa !14
  %3 = tail call spir_func <4 x i32> @_Z21get_sub_group_ge_maskv() #8
  %4 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 1
  store <4 x i32> %3, <4 x i32> addrspace(1)* %4, align 16, !tbaa !14
  %5 = tail call spir_func <4 x i32> @_Z21get_sub_group_gt_maskv() #8
  %6 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 2
  store <4 x i32> %5, <4 x i32> addrspace(1)* %6, align 16, !tbaa !14
  %7 = tail call spir_func <4 x i32> @_Z21get_sub_group_le_maskv() #8
  %8 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 3
  store <4 x i32> %7, <4 x i32> addrspace(1)* %8, align 16, !tbaa !14
  %9 = tail call spir_func <4 x i32> @_Z21get_sub_group_lt_maskv() #8
  %10 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %0, i64 4
  store <4 x i32> %9, <4 x i32> addrspace(1)* %10, align 16, !tbaa !14
  ret void
}

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func <4 x i32> @_Z21get_sub_group_eq_maskv() local_unnamed_addr #5

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func <4 x i32> @_Z21get_sub_group_ge_maskv() local_unnamed_addr #5

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func <4 x i32> @_Z21get_sub_group_gt_maskv() local_unnamed_addr #5

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func <4 x i32> @_Z21get_sub_group_le_maskv() local_unnamed_addr #5

; Function Attrs: convergent nounwind readnone
declare dso_local spir_func <4 x i32> @_Z21get_sub_group_lt_maskv() local_unnamed_addr #5

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="256" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="512" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="1024" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { convergent nofree nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { convergent nounwind }
attributes #8 = { convergent nounwind readnone }

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
!6 = !{!"uint*"}
!7 = !{!""}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"uint4*"}
!13 = !{!"uint __attribute__((ext_vector_type(4)))*"}
!14 = !{!10, !10, i64 0}
