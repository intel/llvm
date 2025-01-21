; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-use-after-return=never -asan-stack-dynamic-alloca=0 -asan-mapping-scale=4 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%struct.Input = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

; Function Attrs: sanitize_address
define spir_func void @test(ptr addrspace(4) byval(%struct.Input) %input) #0 {
entry:
  ; CHECK: inttoptr i64 %1 to ptr addrspace(4)
  ret void
}

attributes #0 = { sanitize_address }
