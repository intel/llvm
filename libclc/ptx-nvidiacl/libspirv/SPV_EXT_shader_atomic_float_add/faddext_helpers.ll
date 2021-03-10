#if __clang_major__ >= 7
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"
#else
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
#endif

define float @__clc__atomic_fetch_add_float_global_relaxed(float addrspace(1)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(1)* %ptr, float %value monotonic
  ret float %0
}

define float @__clc__atomic_fetch_add_float_global_acquire(float addrspace(1)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(1)* %ptr, float %value acquire
  ret float %0
}

define float @__clc__atomic_fetch_add_float_global_release(float addrspace(1)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(1)* %ptr, float %value release
  ret float %0
}

define float @__clc__atomic_fetch_add_float_global_acq_rel(float addrspace(1)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(1)* %ptr, float %value acq_rel
  ret float %0
}

define float @__clc__atomic_fetch_add_float_global_seq_cst(float addrspace(1)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(1)* %ptr, float %value seq_cst
  ret float %0
}

define float @__clc__atomic_fetch_add_float_local_relaxed(float addrspace(3)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(3)* %ptr, float %value monotonic
  ret float %0
}

define float @__clc__atomic_fetch_add_float_local_acquire(float addrspace(3)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(3)* %ptr, float %value acquire
  ret float %0
}

define float @__clc__atomic_fetch_add_float_local_release(float addrspace(3)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(3)* %ptr, float %value release
  ret float %0
}

define float @__clc__atomic_fetch_add_float_local_acq_rel(float addrspace(3)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(3)* %ptr, float %value acq_rel
  ret float %0
}

define float @__clc__atomic_fetch_add_float_local_seq_cst(float addrspace(3)* nocapture %ptr, float %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd float addrspace(3)* %ptr, float %value seq_cst
  ret float %0
}

define double @__clc__atomic_fetch_add_double_global_relaxed(double addrspace(1)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(1)* %ptr, double %value monotonic
  ret double %0
}

define double @__clc__atomic_fetch_add_double_global_acquire(double addrspace(1)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(1)* %ptr, double %value acquire
  ret double %0
}

define double @__clc__atomic_fetch_add_double_global_release(double addrspace(1)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(1)* %ptr, double %value release
  ret double %0
}

define double @__clc__atomic_fetch_add_double_global_acq_rel(double addrspace(1)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(1)* %ptr, double %value acq_rel
  ret double %0
}

define double @__clc__atomic_fetch_add_double_global_seq_cst(double addrspace(1)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(1)* %ptr, double %value seq_cst
  ret double %0
}

define double @__clc__atomic_fetch_add_double_local_relaxed(double addrspace(3)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(3)* %ptr, double %value monotonic
  ret double %0
}

define double @__clc__atomic_fetch_add_double_local_acquire(double addrspace(3)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(3)* %ptr, double %value acquire
  ret double %0
}

define double @__clc__atomic_fetch_add_double_local_release(double addrspace(3)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(3)* %ptr, double %value release
  ret double %0
}

define double @__clc__atomic_fetch_add_double_local_acq_rel(double addrspace(3)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(3)* %ptr, double %value acq_rel
  ret double %0
}

define double @__clc__atomic_fetch_add_double_local_seq_cst(double addrspace(3)* nocapture %ptr, double %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile fadd double addrspace(3)* %ptr, double %value seq_cst
  ret double %0
}
