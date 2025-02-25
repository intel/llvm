; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: mustprogress nounwind sanitize_address uwtable
define spir_func void @no_return(ptr addrspace(4) noundef align 8 dereferenceable_or_null(12) %this) unnamed_addr #0 {
  ; CHECK-LABEL:  @no_return
  ; CHECK-NOT: call void @__asan_handle_no_return
  tail call void @llvm.trap() #1
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #2

attributes #0 = { mustprogress nounwind sanitize_address uwtable }
attributes #1 = { noreturn nounwind }
attributes #2 = { cold noreturn nounwind memory(inaccessiblemem: write) }
