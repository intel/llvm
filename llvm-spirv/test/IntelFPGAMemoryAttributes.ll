; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -o %t.spv
; RUN: llvm-spirv %t.spv --spirv-ext=+SPV_INTEL_fpga_memory_attributes -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text -r %t.spt -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
;
; TODO: add a bunch of different tests for --spirv-ext option

; CHECK-SPIRV: Capability FPGAMemoryAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_memory_attributes"
; CHECK-SPIRV: Decorate {{[0-9]+}} MemoryINTEL "DEFAULT"
; CHECK-SPIRV: Decorate {{[0-9]+}} RegisterINTEL
; CHECK-SPIRV: Decorate {{[0-9]+}} MemoryINTEL "BLOCK_RAM"
; CHECK-SPIRV: Decorate {{[0-9]+}} NumbanksINTEL 2
; CHECK-SPIRV: Decorate {{[0-9]+}} NumbanksINTEL 4
; CHECK-SPIRV: Decorate {{[0-9]+}} BankwidthINTEL 8
; CHECK-SPIRV: Decorate {{[0-9]+}} MaxPrivateCopiesINTEL 4
; CHECK-SPIRV: Decorate {{[0-9]+}} SinglepumpINTEL
; CHECK-SPIRV: Decorate {{[0-9]+}} DoublepumpINTEL
; CHECK-SPIRV: Decorate {{[0-9]+}} MaxReplicatesINTEL 2
; CHECK-SPIRV: Decorate {{[0-9]+}} SimpleDualPortINTEL
; CHECK-SPIRV: Decorate {{[0-9]+}} MergeINTEL "foo" "depth"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

%class.anon = type { i8 }
%struct._ZTS7foo_two.foo_two = type { i32 }

; CHECK-LLVM: [[STR1:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{numbanks:4}
; CHECK-LLVM: [[STR2:@[0-9_.]+]] = {{.*}}{register:1}
; CHECK-LLVM: [[STR3:@[0-9_.]+]] = {{.*}}{memory:BLOCK_RAM}
; CHECK-LLVM: [[STR4:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{bankwidth:8}
; CHECK-LLVM: [[STR5:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{max_private_copies:4}
; CHECK-LLVM: [[STR6:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{pump:1}
; CHECK-LLVM: [[STR7:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{pump:2}
; CHECK-LLVM: [[STR8:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{merge:foo:depth}
; CHECK-LLVM: [[STR9:@[0-9_.]+]] = {{.*}}{max_replicates:2}
; CHECK-LLVM: [[STR10:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{simple_dual_port:1}
; CHECK-LLVM: [[STR11:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{numbanks:2}
@.str = private unnamed_addr constant [29 x i8] c"{memory:DEFAULT}{numbanks:4}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [13 x i8] c"test_var.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [13 x i8] c"{register:1}\00", section "llvm.metadata"
@.str.3 = private unnamed_addr constant [19 x i8] c"{memory:BLOCK_RAM}\00", section "llvm.metadata"
@.str.4 = private unnamed_addr constant [30 x i8] c"{memory:DEFAULT}{bankwidth:8}\00", section "llvm.metadata"
@.str.5 = private unnamed_addr constant [39 x i8] c"{memory:DEFAULT}{max_private_copies:4}\00", section "llvm.metadata"
@.str.6 = private unnamed_addr constant [25 x i8] c"{memory:DEFAULT}{pump:1}\00", section "llvm.metadata"
@.str.7 = private unnamed_addr constant [25 x i8] c"{memory:DEFAULT}{pump:2}\00", section "llvm.metadata"
@.str.8 = private unnamed_addr constant [34 x i8] c"{memory:DEFAULT}{merge:foo:depth}\00", section "llvm.metadata"
@.str.9 = private unnamed_addr constant [19 x i8] c"{max_replicates:2}\00", section "llvm.metadata"
@.str.10 = private unnamed_addr constant [37 x i8] c"{memory:DEFAULT}{simple_dual_port:1}\00", section "llvm.metadata"
@.str.11 = private unnamed_addr constant [29 x i8] c"{memory:DEFAULT}{numbanks:2}\00", section "llvm.metadata"

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %class.anon, align 1
  %1 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %0)
  %2 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%class.anon* %this) #2 align 2 {
entry:
  %this.addr = alloca %class.anon*, align 8
  store %class.anon* %this, %class.anon** %this.addr, align 8, !tbaa !5
  %this1 = load %class.anon*, %class.anon** %this.addr, align 8
  call spir_func void @_Z3foov()
  call spir_func void @_Z3boov()
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
define spir_func void @_Z3foov() #3 {
entry:
  %var_one = alloca i32, align 4
  %var_two = alloca i32, align 4
  %var_three = alloca i32, align 4
  %var_four = alloca i32, align 4
  %var_five = alloca i8, align 1
  %var_six = alloca i32, align 4
  %var_seven = alloca i32, align 4
  %var_eight = alloca i32, align 4
  %var_nine = alloca i32, align 4
  %var_ten = alloca i32, align 4
  %0 = bitcast i32* %var_one to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  %var_one1 = bitcast i32* %var_one to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* %[[VAR1:[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([29 x i8], [29 x i8]* [[STR1]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_one1, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 2)
  %1 = bitcast i32* %var_two to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #4
  %var_two2 = bitcast i32* %var_two to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR2:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([13 x i8], [13 x i8]* [[STR2]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_two2, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 3)
  %2 = bitcast i32* %var_three to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %2) #4
  %var_three3 = bitcast i32* %var_three to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR3:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[STR3]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_three3, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 4)
  %3 = bitcast i32* %var_four to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #4
  %var_four4 = bitcast i32* %var_four to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR4:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([30 x i8], [30 x i8]* [[STR4]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_four4, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 5)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %var_five) #4
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR5:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([39 x i8], [39 x i8]* [[STR5]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_five, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 6)
  %4 = bitcast i32* %var_six to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #4
  %var_six6 = bitcast i32* %var_six to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR6:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([25 x i8], [25 x i8]* [[STR6]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_six6, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.6, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 7)
  %5 = bitcast i32* %var_seven to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #4
  %var_seven7 = bitcast i32* %var_seven to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR7:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([25 x i8], [25 x i8]* [[STR7]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_seven7, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.7, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 8)
  %6 = bitcast i32* %var_eight to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #4
  %var_eight8 = bitcast i32* %var_eight to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR8:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([34 x i8], [34 x i8]* [[STR8]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_eight8, i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.8, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 9)
  %7 = bitcast i32* %var_nine to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #4
  %var_nine9 = bitcast i32* %var_nine to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR9:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([19 x i8], [19 x i8]* [[STR9]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_nine9, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.9, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 10)
  %8 = bitcast i32* %var_ten to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %8) #4
  %var_ten10 = bitcast i32* %var_ten to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* [[VAR10:%[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([37 x i8], [37 x i8]* [[STR10]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %var_ten10, i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.10, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 11)
  %9 = bitcast i32* %var_ten to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %9) #4
  %10 = bitcast i32* %var_nine to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %10) #4
  %11 = bitcast i32* %var_eight to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %11) #4
  %12 = bitcast i32* %var_seven to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %12) #4
  %13 = bitcast i32* %var_six to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %13) #4
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %var_five) #4
  %14 = bitcast i32* %var_four to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %14) #4
  %15 = bitcast i32* %var_three to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %15) #4
  %16 = bitcast i32* %var_two to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %16) #4
  %17 = bitcast i32* %var_one to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %17) #4
  ret void
}

; Function Attrs: nounwind
define dso_local spir_func void @_Z3boov() #3 {
  %1 = alloca %struct._ZTS7foo_two.foo_two, align 4
  %2 = bitcast %struct._ZTS7foo_two.foo_two* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 44, i8* %2) #4
  %3 = bitcast %struct._ZTS7foo_two.foo_two* %1 to i8*
  ; CHECK-LLVM: call void @llvm.var.annotation(i8* %[[VAR11:[a-zA-Z0-9_]+]], i8* getelementptr inbounds ([29 x i8], [29 x i8]* [[STR11]], i32 0, i32 0), i8* undef, i32 undef)
  call void @llvm.var.annotation(i8* %3, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.11, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.1, i32 0, i32 0), i32 35)
   %4 = bitcast %struct._ZTS7foo_two.foo_two* %1 to i8*
   call void @llvm.lifetime.end.p0i8(i64 44, i8* %4) #4
   ret void
}

; Function Attrs: nounwind
declare void @llvm.var.annotation(i8*, i8*, i8*, i32) #4

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind optnone noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 9.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
