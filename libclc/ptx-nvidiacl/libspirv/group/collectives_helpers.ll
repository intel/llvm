; 64 storage locations is sufficient for all current-generation NVIDIA GPUs
; 64 bits per warp is sufficient for all fundamental data types
; Reducing storage for small data types or increasing it for user-defined types
; will likely require an additional pass to track group algorithm usage
@__clc__group_scratch = internal addrspace(3) global [64 x i64] undef, align 1

define i8 addrspace(3)* @__clc__get_group_scratch_bool() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to i8 addrspace(3)*
  ret i8 addrspace(3)* %cast
}

define i8 addrspace(3)* @__clc__get_group_scratch_char() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to i8 addrspace(3)*
  ret i8 addrspace(3)* %cast
}

define i16 addrspace(3)* @__clc__get_group_scratch_short() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to i16 addrspace(3)*
  ret i16 addrspace(3)* %cast
}

define i32 addrspace(3)* @__clc__get_group_scratch_int() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to i32 addrspace(3)*
  ret i32 addrspace(3)* %cast
}

define i64 addrspace(3)* @__clc__get_group_scratch_long() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to i64 addrspace(3)*
  ret i64 addrspace(3)* %cast
}

define half addrspace(3)* @__clc__get_group_scratch_half() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to half addrspace(3)*
  ret half addrspace(3)* %cast
}

define float addrspace(3)* @__clc__get_group_scratch_float() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to float addrspace(3)*
  ret float addrspace(3)* %cast
}

define double addrspace(3)* @__clc__get_group_scratch_double() nounwind alwaysinline {
entry:
  %ptr = getelementptr inbounds [64 x i64], [64 x i64] addrspace(3)* @__clc__group_scratch, i64 0, i64 0
  %cast = bitcast i64 addrspace(3)* %ptr to double addrspace(3)*
  ret double addrspace(3)* %cast
}
