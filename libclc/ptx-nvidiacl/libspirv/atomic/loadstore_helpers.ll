define i32 @__clc__atomic_load_global_4_unordered(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(1)* %ptr unordered, align 4
  ret i32 %0
}

define i32 @__clc__atomic_load_local_4_unordered(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(3)* %ptr unordered, align 4
  ret i32 %0
}

define i64 @__clc__atomic_load_global_8_unordered(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(1)* %ptr unordered, align 8
  ret i64 %0
}

define i64 @__clc__atomic_load_local_8_unordered(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(3)* %ptr unordered, align 8
  ret i64 %0
}

define i32 @__clc__atomic_uload_global_4_unordered(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(1)* %ptr unordered, align 4
  ret i32 %0
}

define i32 @__clc__atomic_uload_local_4_unordered(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(3)* %ptr unordered, align 4
  ret i32 %0
}

define i64 @__clc__atomic_uload_global_8_unordered(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(1)* %ptr unordered, align 8
  ret i64 %0
}

define i64 @__clc__atomic_uload_local_8_unordered(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(3)* %ptr unordered, align 8
  ret i64 %0
}

define i32 @__clc__atomic_load_global_4_acquire(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(1)* %ptr acquire, align 4
  ret i32 %0
}

define i32 @__clc__atomic_load_local_4_acquire(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(3)* %ptr acquire, align 4
  ret i32 %0
}

define i64 @__clc__atomic_load_global_8_acquire(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(1)* %ptr acquire, align 8
  ret i64 %0
}

define i64 @__clc__atomic_load_local_8_acquire(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(3)* %ptr acquire, align 8
  ret i64 %0
}

define i32 @__clc__atomic_uload_global_4_acquire(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(1)* %ptr acquire, align 4
  ret i32 %0
}

define i32 @__clc__atomic_uload_local_4_acquire(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(3)* %ptr acquire, align 4
  ret i32 %0
}

define i64 @__clc__atomic_uload_global_8_acquire(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(1)* %ptr acquire, align 8
  ret i64 %0
}

define i64 @__clc__atomic_uload_local_8_acquire(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(3)* %ptr acquire, align 8
  ret i64 %0
}


define i32 @__clc__atomic_load_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(1)* %ptr seq_cst, align 4
  ret i32 %0
}

define i32 @__clc__atomic_load_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(3)* %ptr seq_cst, align 4
  ret i32 %0
}

define i64 @__clc__atomic_load_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(1)* %ptr seq_cst, align 8
  ret i64 %0
}

define i64 @__clc__atomic_load_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(3)* %ptr seq_cst, align 8
  ret i64 %0
}

define i32 @__clc__atomic_uload_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(1)* %ptr seq_cst, align 4
  ret i32 %0
}

define i32 @__clc__atomic_uload_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i32, i32 addrspace(3)* %ptr seq_cst, align 4
  ret i32 %0
}

define i64 @__clc__atomic_uload_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(1)* %ptr seq_cst, align 8
  ret i64 %0
}

define i64 @__clc__atomic_uload_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr) nounwind alwaysinline {
entry:
  %0 = load atomic volatile i64, i64 addrspace(3)* %ptr seq_cst, align 8
  ret i64 %0
}

define void @__clc__atomic_store_global_4_unordered(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(1)* %ptr unordered, align 4
  ret void
}

define void @__clc__atomic_store_local_4_unordered(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(3)* %ptr unordered, align 4
  ret void
}

define void @__clc__atomic_store_global_8_unordered(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(1)* %ptr unordered, align 8
  ret void
}

define void @__clc__atomic_store_local_8_unordered(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(3)* %ptr unordered, align 8
  ret void
}

define void @__clc__atomic_ustore_global_4_unordered(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(1)* %ptr unordered, align 4
  ret void
}

define void @__clc__atomic_ustore_local_4_unordered(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(3)* %ptr unordered, align 4
  ret void
}

define void @__clc__atomic_ustore_global_8_unordered(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(1)* %ptr unordered, align 8
  ret void
}

define void @__clc__atomic_ustore_local_8_unordered(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(3)* %ptr unordered, align 8
  ret void
}

define void @__clc__atomic_store_global_4_release(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(1)* %ptr release, align 4
  ret void
}

define void @__clc__atomic_store_local_4_release(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(3)* %ptr release, align 4
  ret void
}

define void @__clc__atomic_store_global_8_release(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(1)* %ptr release, align 8
  ret void
}

define void @__clc__atomic_store_local_8_release(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(3)* %ptr release, align 8
  ret void
}

define void @__clc__atomic_ustore_global_4_release(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(1)* %ptr release, align 4
  ret void
}

define void @__clc__atomic_ustore_local_4_release(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(3)* %ptr release, align 4
  ret void
}

define void @__clc__atomic_ustore_global_8_release(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(1)* %ptr release, align 8
  ret void
}

define void @__clc__atomic_ustore_local_8_release(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(3)* %ptr release, align 8
  ret void
}

define void @__clc__atomic_store_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(1)* %ptr seq_cst, align 4
  ret void
}

define void @__clc__atomic_store_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(3)* %ptr seq_cst, align 4
  ret void
}

define void @__clc__atomic_store_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(1)* %ptr seq_cst, align 8
  ret void
}

define void @__clc__atomic_store_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(3)* %ptr seq_cst, align 8
  ret void
}

define void @__clc__atomic_ustore_global_4_seq_cst(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(1)* %ptr seq_cst, align 4
  ret void
}

define void @__clc__atomic_ustore_local_4_seq_cst(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  store atomic volatile i32 %value, i32 addrspace(3)* %ptr seq_cst, align 4
  ret void
}

define void @__clc__atomic_ustore_global_8_seq_cst(i64 addrspace(1)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(1)* %ptr seq_cst, align 8
  ret void
}

define void @__clc__atomic_ustore_local_8_seq_cst(i64 addrspace(3)* nocapture %ptr, i64 %value) nounwind alwaysinline {
entry:
  store atomic volatile i64 %value, i64 addrspace(3)* %ptr seq_cst, align 8
  ret void
}
