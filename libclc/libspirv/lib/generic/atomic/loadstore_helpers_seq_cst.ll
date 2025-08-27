; This file contains helper functions for the seq_cst memory ordering constraint.
; Other targets can specialize this file to account for unsupported features in their backend.

declare void @llvm.trap()

define i32 @__clc__atomic_load_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i32 @__clc__atomic_load_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i32 @__clc__atomic_load__4_seq_cst(i32 addrspace(0)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_load_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_load_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_load__8_seq_cst(i64 addrspace(0)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i32 @__clc__atomic_uload_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(1)* %ptr seq_cst, align 4
  ret i32 %0
}

define i32 @__clc__atomic_uload_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i32 @__clc__atomic_uload__4_seq_cst(i32 addrspace(0)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_uload_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_uload_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_uload__8_seq_cst(i64 addrspace(0)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store__4_seq_cst(i32 addrspace(0)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store__8_seq_cst(i64 addrspace(0)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore__4_seq_cst(i32 addrspace(0)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore__8_seq_cst(i64 addrspace(0)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}
