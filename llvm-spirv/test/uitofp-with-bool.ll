; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r -spirv-target-env="SPV-IR" %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=SPV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=LLVM
; RUN: spirv-val %t.spv

; The IR was generated from the following source:
; void __kernel K(global float* A, int B) {
;   bool Cmp = B > 0;
;   A[0] = Cmp;
; }
; Command line:
; clang -x cl -cl-std=CL2.0 -target spir64 -emit-llvm -S -c test.cl


; SPV-DAG: Name [[s1:[0-9]+]] "s1"
; SPV-DAG: Name [[s2:[0-9]+]] "s2"
; SPV-DAG: Name [[s3:[0-9]+]] "s3"
; SPV-DAG: Name [[s4:[0-9]+]] "s4"
; SPV-DAG: Name [[s5:[0-9]+]] "s5"
; SPV-DAG: Name [[s6:[0-9]+]] "s6"
; SPV-DAG: Name [[s7:[0-9]+]] "s7"
; SPV-DAG: Name [[s8:[0-9]+]] "s8"
; SPV-DAG: Name [[z1:[0-9]+]] "z1"
; SPV-DAG: Name [[z2:[0-9]+]] "z2"
; SPV-DAG: Name [[z3:[0-9]+]] "z3"
; SPV-DAG: Name [[z4:[0-9]+]] "z4"
; SPV-DAG: Name [[z5:[0-9]+]] "z5"
; SPV-DAG: Name [[z6:[0-9]+]] "z6"
; SPV-DAG: Name [[z7:[0-9]+]] "z7"
; SPV-DAG: Name [[z8:[0-9]+]] "z8"
; SPV-DAG: Name [[ufp1:[0-9]+]] "ufp1"
; SPV-DAG: Name [[ufp2:[0-9]+]] "ufp2"
; SPV-DAG: Name [[sfp1:[0-9]+]] "sfp1"
; SPV-DAG: Name [[sfp2:[0-9]+]] "sfp2"
; SPV-DAG: TypeInt [[int_32:[0-9]+]] 32 0
; SPV-DAG: TypeInt [[int_8:[0-9]+]] 8 0
; SPV-DAG: TypeInt [[int_16:[0-9]+]] 16 0
; SPV-DAG: TypeInt [[int_64:[0-9]+]] 64 0
; SPV-DAG: Constant [[int_32]] [[zero_32:[0-9]+]] 0
; SPV-DAG: Constant [[int_32]] [[one_32:[0-9]+]] 1
; SPV-DAG: Constant [[int_8]] [[zero_8:[0-9]+]] 0
; SPV-DAG: Constant [[int_8]] [[mone_8:[0-9]+]] 255
; SPV-DAG: Constant [[int_16]] [[zero_16:[0-9]+]] 0
; SPV-DAG: Constant [[int_16]] [[mone_16:[0-9]+]] 65535
; SPV-DAG: Constant [[int_32]] [[mone_32:[0-9]+]] 4294967295
; SPV-DAG: Constant [[int_64]] [[zero_64:[0-9]+]] 0 0
; SPV-DAG: Constant [[int_64]] [[mone_64:[0-9]+]] 4294967295 4294967295
; SPV-DAG: Constant [[int_8]] [[one_8:[0-9]+]] 1
; SPV-DAG: Constant [[int_16]] [[one_16:[0-9]+]] 1
; SPV-DAG: Constant [[int_64]] [[one_64:[0-9]+]] 1 0
; SPV-DAG: TypeVoid [[void:[0-9]+]]
; SPV-DAG: TypeFloat [[float:[0-9]+]] 32
; SPV-DAG: TypeBool [[bool:[0-9]+]]
; SPV-DAG: TypeVector [[vec_8:[0-9]+]] [[int_8]] 2
; SPV-DAG: TypeVector [[vec_1:[0-9]+]] [[bool]] 2
; SPV-DAG: TypeVector [[vec_16:[0-9]+]] [[int_16]] 2
; SPV-DAG: TypeVector [[vec_32:[0-9]+]] [[int_32]] 2
; SPV-DAG: TypeVector [[vec_64:[0-9]+]] [[int_64]] 2
; SPV-DAG: TypeVector [[vec_float:[0-9]+]] [[float]] 2
; SPV-DAG: ConstantNull [[vec_8]] [[zeros_8:[0-9]+]]
; SPV-DAG: ConstantComposite [[vec_8]] [[mones_8:[0-9]+]] [[mone_8]] [[mone_8]]
; SPV-DAG: ConstantNull [[vec_16]] [[zeros_16:[0-9]+]]
; SPV-DAG: ConstantComposite [[vec_16]] [[mones_16:[0-9]+]] [[mone_16]] [[mone_16]]
; SPV-DAG: ConstantNull [[vec_32]] [[zeros_32:[0-9]+]]
; SPV-DAG: ConstantComposite [[vec_32]] [[mones_32:[0-9]+]] [[mone_32]] [[mone_32]]
; SPV-DAG: ConstantNull [[vec_64]] [[zeros_64:[0-9]+]]
; SPV-DAG: ConstantComposite [[vec_64]] [[mones_64:[0-9]+]] [[mone_64]] [[mone_64]]
; SPV-DAG: ConstantComposite [[vec_8]] [[ones_8:[0-9]+]] [[one_8]] [[one_8]]
; SPV-DAG: ConstantComposite [[vec_16]] [[ones_16:[0-9]+]] [[one_16]] [[one_16]]
; SPV-DAG: ConstantComposite [[vec_32]] [[ones_32:[0-9]+]] [[one_32]] [[one_32]]
; SPV-DAG: ConstantComposite [[vec_64]] [[ones_64:[0-9]+]] [[one_64]] [[one_64]]


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; SPV-DAG: Function
; SPV-DAG: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
; SPV-DAG: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
; SPV-DAG: FunctionParameter {{[0-9]+}} [[i1s:[0-9]+]]
; SPV-DAG: FunctionParameter {{[0-9]+}} [[i1v:[0-9]+]]

; Function Attrs: nofree norecurse nounwind writeonly
define dso_local spir_kernel void @K(float addrspace(1)* nocapture %A, i32 %B, i1 %i1s, <2 x i1> %i1v) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:


; SPV-DAG: SGreaterThan [[bool]] [[cmp_res:[0-9]+]] [[B]] [[zero_32]]
; LLVM-DAG: %cmp = icmp sgt i32 %B, 0
  %cmp = icmp sgt i32 %B, 0
; SPV-DAG: Select [[int_32]] [[select_res:[0-9]+]] [[cmp_res]] [[one_32]] [[zero_32]]
; SPV-DAG: ConvertUToF [[float]] [[utof_res:[0-9]+]] [[select_res]]
; LLVM-DAG: %[[sel_res_0:[0-9]+]] = select i1 %cmp, i32 1, i32 0
; LLVM-DAG: %conv = uitofp i32 %[[sel_res_0]] to float
  %conv = uitofp i1 %cmp to float
; SPV-DAG: Store [[A]] [[utof_res]]
; LLVM-DAG: store float %conv, float addrspace(1)* %A, align 4
  store float %conv, float addrspace(1)* %A, align 4;

; SPV-DAG: Select [[int_8]] [[s1]] [[i1s]] [[mone_8]] [[zero_8]]
; LLVM-DAG: %s1 = select i1 %i1s, i8 -1, i8 0
  %s1 = sext i1 %i1s to i8
; SPV-DAG: Select [[int_16]] [[s2]] [[i1s]] [[mone_16]] [[zero_16]]
; LLVM-DAG: %s2 = select i1 %i1s, i16 -1, i16 0
  %s2 = sext i1 %i1s to i16
; SPV-DAG: Select [[int_32]] [[s3]] [[i1s]] [[mone_32]] [[zero_32]]
; LLVM-DAG: %s3 = select i1 %i1s, i32 -1, i32 0
  %s3 = sext i1 %i1s to i32
; SPV-DAG: Select [[int_64]] [[s4]] [[i1s]] [[mone_64]] [[zero_64]]
; LLVM-DAG: %s4 = select i1 %i1s, i64 -1, i64 0
  %s4 = sext i1 %i1s to i64
; SPV-DAG: Select [[vec_8]] [[s5]] [[i1v]] [[mones_8]] [[zeros_8]]
; LLVM-DAG: %s5 = select <2 x i1> %i1v, <2 x i8> <i8 -1, i8 -1>, <2 x i8> zeroinitializer
  %s5 = sext <2 x i1> %i1v to <2 x i8>
; SPV-DAG: Select [[vec_16]] [[s6]] [[i1v]] [[mones_16]] [[zeros_16]]
; LLVM-DAG: %s6 = select <2 x i1> %i1v, <2 x i16> <i16 -1, i16 -1>, <2 x i16> zeroinitializer
  %s6 = sext <2 x i1> %i1v to <2 x i16>
; SPV-DAG: Select [[vec_32]] [[s7]] [[i1v]] [[mones_32]] [[zeros_32]]
; LLVM-DAG: %s7 = select <2 x i1> %i1v, <2 x i32> <i32 -1, i32 -1>, <2 x i32> zeroinitializer
  %s7 = sext <2 x i1> %i1v to <2 x i32>
; SPV-DAG: Select [[vec_64]] [[s8]] [[i1v]] [[mones_64]] [[zeros_64]]
; LLVM-DAG: %s8 = select <2 x i1> %i1v, <2 x i64> <i64 -1, i64 -1>, <2 x i64> zeroinitializer
  %s8 = sext <2 x i1> %i1v to <2 x i64>
; SPV-DAG: Select [[int_8]] [[z1]] [[i1s]] [[one_8]] [[zero_8]]
; LLVM-DAG: %z1 = select i1 %i1s, i8 1, i8 0
  %z1 = zext i1 %i1s to i8
; SPV-DAG: Select [[int_16]] [[z2]] [[i1s]] [[one_16]] [[zero_16]]
; LLVM-DAG: %z2 = select i1 %i1s, i16 1, i16 0
  %z2 = zext i1 %i1s to i16
; SPV-DAG: Select [[int_32]] [[z3]] [[i1s]] [[one_32]] [[zero_32]]
; LLVM-DAG: %z3 = select i1 %i1s, i32 1, i32 0
  %z3 = zext i1 %i1s to i32
; SPV-DAG: Select [[int_64]] [[z4]] [[i1s]] [[one_64]] [[zero_64]]
; LLVM-DAG: %z4 = select i1 %i1s, i64 1, i64 0
  %z4 = zext i1 %i1s to i64
; SPV-DAG: Select [[vec_8]] [[z5]] [[i1v]] [[ones_8]] [[zeros_8]]
; LLVM-DAG: %z5 = select <2 x i1> %i1v, <2 x i8> <i8 1, i8 1>, <2 x i8> zeroinitializer
  %z5 = zext <2 x i1> %i1v to <2 x i8>
; SPV-DAG: Select [[vec_16]] [[z6]] [[i1v]] [[ones_16]] [[zeros_16]]
; LLVM-DAG: %z6 = select <2 x i1> %i1v, <2 x i16> <i16 1, i16 1>, <2 x i16> zeroinitializer
  %z6 = zext <2 x i1> %i1v to <2 x i16>
; SPV-DAG: Select [[vec_32]] [[z7]] [[i1v]] [[ones_32]] [[zeros_32]]
; LLVM-DAG: %z7 = select <2 x i1> %i1v, <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer
  %z7 = zext <2 x i1> %i1v to <2 x i32>
; SPV-DAG: Select [[vec_64]] [[z8]] [[i1v]] [[ones_64]] [[zeros_64]]
; LLVM-DAG: %z8 = select <2 x i1> %i1v, <2 x i64> <i64 1, i64 1>, <2 x i64> zeroinitializer
  %z8 = zext <2 x i1> %i1v to <2 x i64>
; SPV-DAG: Select [[int_32]] [[ufp1_res:[0-9]+]] [[i1s]] [[one_32]] [[zero_32]]
; SPV-DAG: ConvertUToF [[float]] [[ufp1]] [[ufp1_res]]
; LLVM-DAG: %[[ufp1_res_llvm:[0-9]+]] = select i1 %i1s, i32 1, i32 0
; LLVM-DAG: %ufp1 = uitofp i32 %[[ufp1_res_llvm]] to float
  %ufp1 = uitofp i1 %i1s to float
; SPV-DAG: Select [[vec_32]] [[ufp2_res:[0-9]+]] [[i1v]] [[ones_32]] [[zeros_32]]
; SPV-DAG: ConvertUToF [[vec_float]] [[ufp2]] [[ufp2_res]]
; LLVM-DAG: %[[ufp2_res_llvm:[0-9]+]] = select <2 x i1> %i1v, <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer
; LLVM-DAG: %ufp2 = uitofp <2 x i32> %[[ufp2_res_llvm]] to <2 x float>
  %ufp2 = uitofp <2 x i1> %i1v to <2 x float>
; SPV-DAG: Select [[int_32]] [[sfp1_res:[0-9]+]] [[i1s]] [[one_32]] [[zero_32]]
; SPV-DAG: ConvertSToF [[float]] [[sfp1]] [[sfp1_res]]
; LLVM-DAG: %[[sfp1_res_llvm:[0-9]+]] = select i1 %i1s, i32 1, i32 0
; LLVM-DAG: %sfp1 = sitofp i32 %[[sfp1_res_llvm:[0-9]+]] to float
  %sfp1 = sitofp i1 %i1s to float
; SPV-DAG: Select [[vec_32]] [[sfp2_res:[0-9]+]] [[i1v]] [[ones_32]] [[zeros_32]]
; SPV-DAG: ConvertSToF [[vec_float]] [[sfp2]] [[sfp2_res]]
; LLVM-DAG: %[[sfp2_res_llvm:[0-9]+]] = select <2 x i1> %i1v, <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer
; LLVM-DAG: %sfp2 = sitofp <2 x i32> %[[sfp2_res_llvm]] to <2 x float>
  %sfp2 = sitofp <2 x i1> %i1v to <2 x float>
  ret void
}


attributes #0 = { nofree norecurse nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{i32 1, i32 0}
!3 = !{!"none", !"none"}
!4 = !{!"float*", !"int"}
!5 = !{!"", !""}
