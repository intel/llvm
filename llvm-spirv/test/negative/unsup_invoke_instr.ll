;Translator does not parse some llvm instructions
;and emit errror message in that case.
; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s

; CHECK: InvalidInstruction: Can't translate llvm instruction:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE6Kernel" = comdat any


; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE6Kernel"() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !4 {
entry:
  ret void
}

; Function Attrs: noinline optnone uwtable
define dso_local i32 @_funcEx() #4 {
  ret i32 0
}

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @func() #5 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = invoke i32 @_funcEx()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { i8*, i32 }
          cleanup
  ret i32 0
}

declare dso_local i32 @__gxx_personality_v0(...)

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/localdisk2/icl/fadeeval/sycl_workspace/reproducer2/test.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0 (https://github.com/intel/llvm e6d68c098feda598ef8030ef211076f899eff93a)"}
!4 = !{}
