#if __clang_major__ >= 7
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
#else
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
#endif

define i64 @__clc__sync_fetch_and_min_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umin_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_min_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umin_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_min_generic_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umin_generic_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_max_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umax_global_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_max_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umax_local_8(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i64 addrspace(3)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_max_generic_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}

define i64 @__clc__sync_fetch_and_umax_generic_8(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %value seq_cst
  ret i64 %0
}
