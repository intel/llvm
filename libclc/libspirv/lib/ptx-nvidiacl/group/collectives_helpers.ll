; 32 storage locations is sufficient for all current-generation NVIDIA GPUs
@__clc__group_scratch_i1 = internal addrspace(3) global [32 x i1] undef, align 1
@__clc__group_scratch_i8 = internal addrspace(3) global [32 x i8] undef, align 1
@__clc__group_scratch_i16 = internal addrspace(3) global [32 x i16] undef, align 2
@__clc__group_scratch_i32 = internal addrspace(3) global [32 x i32] undef, align 4
@__clc__group_scratch_i64 = internal addrspace(3) global [32 x i64] undef, align 8
@__clc__group_scratch_i128 = internal addrspace(3) global [32 x i128] undef, align 8

define ptr addrspace(3) @__clc__get_group_scratch_bool() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i1], ptr addrspace(3) @__clc__group_scratch_i1, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_char() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i8], ptr addrspace(3) @__clc__group_scratch_i8, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_short() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i16], ptr addrspace(3) @__clc__group_scratch_i16, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define i32 addrspace(3)* @__clc__get_group_scratch_int() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i32], ptr addrspace(3) @__clc__group_scratch_i32, i64 0, i64 0
  ret i32 addrspace(3)* %0
}

define ptr addrspace(3) @__clc__get_group_scratch_long() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i64], ptr addrspace(3) @__clc__group_scratch_i64, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_half() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i16], ptr addrspace(3) @__clc__group_scratch_i16, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_float() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i32], ptr addrspace(3) @__clc__group_scratch_i32, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_double() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i64], ptr addrspace(3) @__clc__group_scratch_i64, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_complex_half() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i32], ptr addrspace(3) @__clc__group_scratch_i32, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_complex_float() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i64], ptr addrspace(3) @__clc__group_scratch_i64, i64 0, i64 0
  ret ptr addrspace(3) %0
}

define ptr addrspace(3) @__clc__get_group_scratch_complex_double() nounwind alwaysinline {
entry:
  %0 = getelementptr inbounds [32 x i128], ptr addrspace(3) @__clc__group_scratch_i128, i64 0, i64 0
  ret ptr addrspace(3) %0
}
