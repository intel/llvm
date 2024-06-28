; ------------------------------------------------
; OCL_asm0418e209feebb096_0045_Unify_after_Inlinerforalways_inlinefunctions.ll
; LLVM major version: 14
; ------------------------------------------------

Printing <null> Function
; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE() #0 !kernel_arg_addr_space !379 !kernel_arg_access_qual !379 !kernel_arg_type !379 !kernel_arg_type_qual !379 !kernel_arg_base_type !379 !kernel_arg_name !379 {
  %1 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 1) #3
  %2 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 1) #3
  %3 = mul i32 %2, %1
  %4 = call spir_func i32 @__builtin_IB_get_local_id_y() #3
  %5 = icmp ult i32 %4, 65536
  call void @llvm.assume(i1 %5) #4
  %6 = add i32 %4, %3
  %7 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 1) #3
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = icmp ult i64 %9, 2147483648
  call void @llvm.assume(i1 %10)
  %11 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 0) #3
  %12 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 0) #3
  %13 = mul i32 %12, %11
  %14 = call spir_func i32 @__builtin_IB_get_local_id_x() #3
  %15 = icmp ult i32 %14, 65536
  call void @llvm.assume(i1 %15) #4
  %16 = add i32 %14, %13
  %17 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 0) #3
  %18 = add i32 %16, %17
  %19 = zext i32 %18 to i64
  %20 = icmp ult i64 %19, 2147483648
  call void @llvm.assume(i1 %20)
  %21 = call spir_func i32 @__builtin_IB_get_local_id_y() #3
  %22 = icmp ult i32 %21, 65536
  call void @llvm.assume(i1 %22) #4
  %23 = zext i32 %21 to i64
  %24 = icmp ult i64 %23, 2147483648
  call void @llvm.assume(i1 %24)
  %25 = call spir_func i32 @__builtin_IB_get_local_id_x() #3
  %26 = icmp ult i32 %25, 65536
  call void @llvm.assume(i1 %26) #4
  %27 = zext i32 %25 to i64
  %28 = icmp ult i64 %27, 2147483648
  call void @llvm.assume(i1 %28)
  ret void
}

Printing <null> Function
