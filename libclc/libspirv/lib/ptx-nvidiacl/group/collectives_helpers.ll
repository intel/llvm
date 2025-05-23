; 32 storage locations is sufficient for all current-generation NVIDIA GPUs
@__clc__group_scratch_i1 = internal addrspace(3) global [32 x i1] poison, align 1
@__clc__group_scratch_i8 = internal addrspace(3) global [32 x i8] poison, align 1
@__clc__group_scratch_i16 = internal addrspace(3) global [32 x i16] poison, align 2
@__clc__group_scratch_i32 = internal addrspace(3) global [32 x i32] poison, align 4
@__clc__group_scratch_i64 = internal addrspace(3) global [32 x i64] poison, align 8
@__clc__group_scratch_i128 = internal addrspace(3) global [32 x i128] poison, align 8

define ptr addrspace(3) @__clc__get_group_scratch_bool() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i1
}

define ptr addrspace(3) @__clc__get_group_scratch_char() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i8
}

define ptr addrspace(3) @__clc__get_group_scratch_short() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i16
}

define i32 addrspace(3)* @__clc__get_group_scratch_int() nounwind alwaysinline {
entry:
  ret i32 addrspace(3)* @__clc__group_scratch_i32
}

define ptr addrspace(3) @__clc__get_group_scratch_long() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i64
}

define ptr addrspace(3) @__clc__get_group_scratch_half() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i16
}

define ptr addrspace(3) @__clc__get_group_scratch_float() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i32
}

define ptr addrspace(3) @__clc__get_group_scratch_double() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i64
}

define ptr addrspace(3) @__clc__get_group_scratch_complex_half() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i32
}

define ptr addrspace(3) @__clc__get_group_scratch_complex_float() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i64
}

define ptr addrspace(3) @__clc__get_group_scratch_complex_double() nounwind alwaysinline {
entry:
  ret ptr addrspace(3) @__clc__group_scratch_i128
}
