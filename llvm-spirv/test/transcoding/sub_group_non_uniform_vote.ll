;; #pragma OPENCL EXTENSION cl_khr_subgroup_non_uniform_vote : enable
;; 
;; kernel void testSubGroupElect(global int* dst){
;; 	dst[0] = sub_group_elect();
;; }
;; 
;; kernel void testSubGroupNonUniformAll(global int* dst){
;; 	dst[0] = sub_group_non_uniform_all(0); 
;; }
;; 
;; kernel void testSubGroupNonUniformAny(global int* dst){
;; 	dst[0] = sub_group_non_uniform_any(0);
;; }
;; 
;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable
;; kernel void testSubGroupNonUniformAllEqual(global int* dst){
;;     {
;;         char v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         uchar v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         short v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         ushort v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         int v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         uint v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         long v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         ulong v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         float v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         half v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;;     {
;;         double v = 0;
;;         dst[0] = sub_group_non_uniform_all_equal( v );
;;     }
;; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ModuleID = 'sub_group_non_uniform_vote.cl'
source_filename = "sub_group_non_uniform_vote.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-SPIRV-DAG: {{[0-9]*}} Capability GroupNonUniform
; CHECK-SPIRV-DAG: {{[0-9]*}} Capability GroupNonUniformVote

; CHECK-SPIRV-DAG: TypeBool  [[bool:[0-9]+]]
; CHECK-SPIRV-DAG: TypeInt   [[char:[0-9]+]]   8  0
; CHECK-SPIRV-DAG: TypeInt   [[short:[0-9]+]]  16 0
; CHECK-SPIRV-DAG: TypeInt   [[int:[0-9]+]]    32 0
; CHECK-SPIRV-DAG: TypeInt   [[long:[0-9]+]]   64 0
; CHECK-SPIRV-DAG: TypeFloat [[half:[0-9]+]]   16
; CHECK-SPIRV-DAG: TypeFloat [[float:[0-9]+]]  32
; CHECK-SPIRV-DAG: TypeFloat [[double:[0-9]+]] 64

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
; CHECK-SPIRV: GroupNonUniformElect [[bool]] {{[0-9]+}} [[ScopeSubgroup]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testSubGroupElect
; CHECK-LLVM: call spir_func i32 @_Z15sub_group_electv()

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testSubGroupElect(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z15sub_group_electv() #2
  store i32 %2, ptr addrspace(1) %0, align 4, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z15sub_group_electv() local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformAll [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[false]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testSubGroupNonUniformAll
; CHECK-LLVM: call spir_func i32 @_Z25sub_group_non_uniform_alli(i32 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testSubGroupNonUniformAll(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z25sub_group_non_uniform_alli(i32 0) #2
  store i32 %2, ptr addrspace(1) %0, align 4, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25sub_group_non_uniform_alli(i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformAny [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[false]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testSubGroupNonUniformAny
; CHECK-LLVM: call spir_func i32 @_Z25sub_group_non_uniform_anyi(i32 {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testSubGroupNonUniformAny(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z25sub_group_non_uniform_anyi(i32 0) #2
  store i32 %2, ptr addrspace(1) %0, align 4, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z25sub_group_non_uniform_anyi(i32) local_unnamed_addr #1

; CHECK-SPIRV-LABEL: 5 Function
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[char_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[short_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[int_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[long_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[float_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[half_0]]
; CHECK-SPIRV: GroupNonUniformAllEqual [[bool]] {{[0-9]+}} [[ScopeSubgroup]] [[double_0]]
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM-LABEL: @testSubGroupNonUniformAllEqual
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equalc(i8 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equalc(i8 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equals(i16 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equals(i16 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equali(i32 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equali(i32 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equall(i64 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equall(i64 {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equalf(float {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equalDh(half {{.*}})
; CHECK-LLVM: call spir_func i32 @_Z31sub_group_non_uniform_all_equald(double {{.*}})

; Function Attrs: convergent nounwind
define dso_local spir_kernel void @testSubGroupNonUniformAllEqual(ptr addrspace(1) nocapture) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
  %2 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalc(i8 signext 0) #2
  store i32 %2, ptr addrspace(1) %0, align 4, !tbaa !7
  %3 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalh(i8 zeroext 0) #2
  store i32 %3, ptr addrspace(1) %0, align 4, !tbaa !7
  %4 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equals(i16 signext 0) #2
  store i32 %4, ptr addrspace(1) %0, align 4, !tbaa !7
  %5 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalt(i16 zeroext 0) #2
  store i32 %5, ptr addrspace(1) %0, align 4, !tbaa !7
  %6 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equali(i32 0) #2
  store i32 %6, ptr addrspace(1) %0, align 4, !tbaa !7
  %7 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalj(i32 0) #2
  store i32 %7, ptr addrspace(1) %0, align 4, !tbaa !7
  %8 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equall(i64 0) #2
  store i32 %8, ptr addrspace(1) %0, align 4, !tbaa !7
  %9 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalm(i64 0) #2
  store i32 %9, ptr addrspace(1) %0, align 4, !tbaa !7
  %10 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalf(float 0.000000e+00) #2
  store i32 %10, ptr addrspace(1) %0, align 4, !tbaa !7
  %11 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equalDh(half 0xH0000) #2
  store i32 %11, ptr addrspace(1) %0, align 4, !tbaa !7
  %12 = tail call spir_func i32 @_Z31sub_group_non_uniform_all_equald(double 0.000000e+00) #2
  store i32 %12, ptr addrspace(1) %0, align 4, !tbaa !7
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalc(i8 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalh(i8 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equals(i16 signext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalt(i16 zeroext) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equali(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalj(i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equall(i64) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalm(i64) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalf(float) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equalDh(half) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z31sub_group_non_uniform_all_equald(double) local_unnamed_addr #1

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
!5 = !{!"int*"}
!6 = !{!""}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
