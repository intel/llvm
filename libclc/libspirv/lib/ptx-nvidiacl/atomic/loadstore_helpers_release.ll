declare void @llvm.trap()

define void @__clc__atomic_store_global_4_release(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store_local_4_release(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store_global_8_release(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_store_local_8_release(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_global_4_release(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_local_4_release(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_global_8_release(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

define void @__clc__atomic_ustore_local_8_release(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  tail call void @llvm.trap()
  unreachable
}

