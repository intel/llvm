extern int __nvvm_reflect_ocl(constant char *);

int __clc_nvvm_reflect_arch() { return __nvvm_reflect_ocl("__CUDA_ARCH"); }

int __clc_nvvm_reflect_ftz() { return __nvvm_reflect_ocl("__CUDA_FTZ"); }
