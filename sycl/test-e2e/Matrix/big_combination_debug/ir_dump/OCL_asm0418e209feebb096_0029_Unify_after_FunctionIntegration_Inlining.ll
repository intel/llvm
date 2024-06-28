; ------------------------------------------------
; OCL_asm0418e209feebb096_0029_Unify_after_FunctionIntegration_Inlining.ll
; LLVM major version: 14
; ------------------------------------------------

Printing <null> Function
; Function Attrs: nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE() #0 !kernel_arg_addr_space !379 !kernel_arg_access_qual !379 !kernel_arg_type !379 !kernel_arg_type_qual !379 !kernel_arg_base_type !379 !kernel_arg_name !379 {
  %1 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 1) #4
  %2 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 1) #4
  %3 = mul i32 %2, %1
  %4 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %5 = icmp ult i32 %4, 65536
  call void @llvm.assume(i1 %5) #5
  %6 = add i32 %4, %3
  %7 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 1) #4
  %8 = add i32 %6, %7
  %9 = zext i32 %8 to i64
  %10 = icmp ult i64 %9, 2147483648
  call void @llvm.assume(i1 %10)
  %11 = call spir_func i32 @__builtin_IB_get_group_id(i32 noundef 0) #4
  %12 = call spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef 0) #4
  %13 = mul i32 %12, %11
  %14 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %15 = icmp ult i32 %14, 65536
  call void @llvm.assume(i1 %15) #5
  %16 = add i32 %14, %13
  %17 = call spir_func i32 @__builtin_IB_get_global_offset(i32 noundef 0) #4
  %18 = add i32 %16, %17
  %19 = zext i32 %18 to i64
  %20 = icmp ult i64 %19, 2147483648
  call void @llvm.assume(i1 %20)
  %21 = call spir_func i32 @__builtin_IB_get_local_id_y() #4
  %22 = icmp ult i32 %21, 65536
  call void @llvm.assume(i1 %22) #5
  %23 = zext i32 %21 to i64
  %24 = icmp ult i64 %23, 2147483648
  call void @llvm.assume(i1 %24)
  %25 = call spir_func i32 @__builtin_IB_get_local_id_x() #4
  %26 = icmp ult i32 %25, 65536
  call void @llvm.assume(i1 %26) #5
  %27 = zext i32 %25 to i64
  %28 = icmp ult i64 %27, 2147483648
  call void @llvm.assume(i1 %28)
  ret void
}

Printing <null> Function
