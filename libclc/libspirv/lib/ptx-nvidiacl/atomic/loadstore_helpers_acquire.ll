declare void @llvm.trap()

define i32 @__clc__atomic_load_global_4_acquire(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i32 @__clc__atomic_load_local_4_acquire(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_load_global_8_acquire(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_load_local_8_acquire(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i32 @__clc__atomic_uload_global_4_acquire(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i32 @__clc__atomic_uload_local_4_acquire(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_uload_global_8_acquire(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define i64 @__clc__atomic_uload_local_8_acquire(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

