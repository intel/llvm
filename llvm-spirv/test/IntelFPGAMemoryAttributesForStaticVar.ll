; Source
; void foo(int a) {
;   static int a_one [[intelfpga::numbanks(2)]];
;   a_one = a_one + a;
; }

; void bar(char b) {
;   static char b_one [[intelfpga::memory("MLAB")]];
;   b_one = b_one + b;
; }

; void baz(int c) {
;   static int c_one[[clang::annotate("foobarbaz")]];
;   c_one = c_one + c;
; }

; template <typename name, typename Func>
; __attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
;   kernelFunc();
; }

; int main() {
;   kernel_single_task<class kernel_function>([]() {
;     foo(128);
;     bar(42);
;     baz(16);
;   });
;   return 0;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability FPGAMemoryAttributesINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_fpga_memory_attributes"
; CHECK-SPIRV: Decorate {{[0-9]+}} UserSemantic "foobarbaz"
; CHECK-SPIRV: Decorate {{[0-9]+}} MemoryINTEL "DEFAULT"
; CHECK-SPIRV: Decorate {{[0-9]+}} MemoryINTEL "MLAB"
; CHECK-SPIRV: Decorate {{[0-9]+}} NumbanksINTEL 2

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

; CHECK-LLVM: [[STR:@[0-9_.]+]] = {{.*}}{memory:DEFAULT}{numbanks:2}
; CHECK-LLVM: [[STR2:@[0-9_.]+]] = {{.*}}{memory:MLAB}
; CHECK-LLVM: [[STR3:@[0-9_.]+]] = {{.*}}foobarbaz
; CHECK-LLVM: @llvm.global.annotations
; CHECK-SAME: _ZZ3fooiE5a_one{{.*}}[[STR]]{{.*}}_ZZ3bariE5b_one{{.*}}[[STR2]]{{.*}}_ZZ3baziE5c_one{{.*}}[[STR3]]
@_ZZ3fooiE5a_one = internal addrspace(1) global i32 0, align 4
@.str = private unnamed_addr constant [29 x i8] c"{memory:DEFAULT}{numbanks:2}\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [9 x i8] c"test.cpp\00", section "llvm.metadata"
@_ZZ3barcE5b_one = internal addrspace(1) global i8 0, align 1
@.str.2 = private unnamed_addr constant [14 x i8] c"{memory:MLAB}\00", section "llvm.metadata"
@_ZZ3baziE5c_one = internal addrspace(1) global i32 0, align 4
@.str.3 = private unnamed_addr constant [10 x i8] c"foobarbaz\00", section "llvm.metadata"
@llvm.global.annotations = appending global [3 x { i8 addrspace(1)*, i8*, i8*, i32 }] [{ i8 addrspace(1)*, i8*, i8*, i32 } { i8 addrspace(1)* bitcast (i32 addrspace(1)* @_ZZ3fooiE5a_one to i8 addrspace(1)*), i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i32 0, i32 0), i32 2 }, { i8 addrspace(1)*, i8*, i8*, i32 } { i8 addrspace(1)* @_ZZ3barcE5b_one, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i32 0, i32 0), i32 7 }, { i8 addrspace(1)*, i8*, i8*, i32 } { i8 addrspace(1)* bitcast (i32 addrspace(1)* @_ZZ3baziE5c_one to i8 addrspace(1)*), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i32 0, i32 0), i32 12 }], section "llvm.metadata"

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  %1 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #4
  %2 = addrspacecast %"class._ZTSZ4mainE3$_0.anon"* %0 to %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %2)
  %3 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this) #2 align 2 {
entry:
  %this.addr = alloca %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*, align 8
  store %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this, %"class._ZTSZ4mainE3$_0.anon" addrspace(4)** %this.addr, align 8, !tbaa !5
  %this1 = load %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*, %"class._ZTSZ4mainE3$_0.anon" addrspace(4)** %this.addr, align 8
  call spir_func void @_Z3fooi(i32 128)
  call spir_func void @_Z3barc(i8 signext 42)
  call spir_func void @_Z3bazi(i32 16)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; CHECK-LLVM: void @_Z3fooi(i32 %a)
; Function Attrs: nounwind
define spir_func void @_Z3fooi(i32 %a) #3 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4, !tbaa !9
  %0 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @_ZZ3fooiE5a_one to i32 addrspace(4)*), align 4, !tbaa !9
  %1 = load i32, i32* %a.addr, align 4, !tbaa !9
  %add = add nsw i32 %0, %1
  store i32 %add, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @_ZZ3fooiE5a_one to i32 addrspace(4)*), align 4, !tbaa !9
  ret void
}

; CHECK-LLVM: void @_Z3barc(i8 signext %b)
; Function Attrs: nounwind
define spir_func void @_Z3barc(i8 signext %b) #3 {
entry:
  %b.addr = alloca i8, align 1
  store i8 %b, i8* %b.addr, align 1, !tbaa !11
  %0 = load i8, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* @_ZZ3barcE5b_one to i8 addrspace(4)*), align 1, !tbaa !11
  %conv = sext i8 %0 to i32
  %1 = load i8, i8* %b.addr, align 1, !tbaa !11
  %conv1 = sext i8 %1 to i32
  %add = add nsw i32 %conv, %conv1
  %conv2 = trunc i32 %add to i8
  store i8 %conv2, i8 addrspace(4)* addrspacecast (i8 addrspace(1)* @_ZZ3barcE5b_one to i8 addrspace(4)*), align 1, !tbaa !11
  ret void
}

; CHECK-LLVM: void @_Z3bazi(i32 %c)
; Function Attrs: nounwind
define spir_func void @_Z3bazi(i32 %c) #3 {
entry:
  %c.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4, !tbaa !9
  %0 = load i32, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @_ZZ3baziE5c_one to i32 addrspace(4)*), align 4, !tbaa !9
  %1 = load i32, i32* %c.addr, align 4, !tbaa !9
  %add = add nsw i32 %0, %1
  store i32 %add, i32 addrspace(4)* addrspacecast (i32 addrspace(1)* @_ZZ3baziE5c_one to i32 addrspace(4)*), align 4, !tbaa !9
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
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
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = !{!7, !7, i64 0}
