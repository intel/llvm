@str = private addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"

declare i32 @__nvvm_reflect(i8*)

define i32 @__clc_nvvm_reflect_arch() alwaysinline {
  %reflect = call i32 @__nvvm_reflect(i8* addrspacecast (i8 addrspace(1)* getelementptr inbounds ([12 x i8], [12 x i8] addrspace(1)* @str, i32 0, i32 0) to i8*))
  ret i32 %reflect
}
