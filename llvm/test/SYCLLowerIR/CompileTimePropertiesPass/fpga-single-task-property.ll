; RUN: opt -passes=compile-time-properties --mtriple=spir64_fpga-unknown-unknown %s -S | FileCheck %s --check-prefix CHECK-FPGA-IR
; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s --check-prefix CHECK-DEFAULT-IR

; CHECK-DEFAULT-IR-NOT: !max_global_work_dim

; CHECK-FPGA-IR-DAG: @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel0"() #0 {{.*}}!max_global_work_dim ![[MaxGlobWorkDim:[0-9]+]]
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE9TheKernel0"() #0 {
entry:
  ret void
}

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="fpga_single_task_property.cpp" "uniform-work-group-size"="true" "sycl-single-task" }

; CHECK-FPGA-IR-DAG: ![[MaxGlobWorkDim]] = !{i32 0}
