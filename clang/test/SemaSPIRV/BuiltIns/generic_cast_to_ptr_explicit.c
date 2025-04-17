// RUN: %clang_cc1 -O1 -triple spirv64 -fsycl-is-device -verify %s -o -
// RUN: %clang_cc1 -O1 -triple spir64 -fsycl-use-spirv-backend-for-spirv-gen -fsycl-is-device -verify %s -o -
// RUN: %clang_cc1 -O1 -triple spir -fsycl-use-spirv-backend-for-spirv-gen -fsycl-is-device -verify %s -o -
// RUN: %clang_cc1 -O1 -triple spirv64 -verify %s -cl-std=CL3.0 -x cl -o -
// RUN: %clang_cc1 -O1 -triple spirv32 -verify %s -cl-std=CL3.0 -x cl -o -

#ifdef __OPENCL_C_VERSION__
#define SYCL_DEVICE
#else
#define SYCL_DEVICE __attribute__((sycl_device))
#endif

SYCL_DEVICE
void test_missing_arguments(int* p) {
  __builtin_spirv_generic_cast_to_ptr_explicit(p); 
  // expected-error@-1 {{too few arguments to function call, expected 2, have 1}}
  __builtin_spirv_generic_cast_to_ptr_explicit(p, 7, p); 
  // expected-error@-1 {{too many arguments to function call, expected 2, have 3}}
}

SYCL_DEVICE
void test_wrong_flag_value(int* p) {
  __builtin_spirv_generic_cast_to_ptr_explicit(p, 14); 
  // expected-error@-1 {{invalid value for storage class argument}}
}

SYCL_DEVICE
void test_wrong_address_space(__attribute__((opencl_local)) int* p) {
  __builtin_spirv_generic_cast_to_ptr_explicit(p, 14); 
  // expected-error@-1 {{expecting a pointer argument to the generic address space}}
}

SYCL_DEVICE
void test_not_a_pointer(int p) {
  __builtin_spirv_generic_cast_to_ptr_explicit(p, 14); 
  // expected-error@-1 {{expecting a pointer argument to the generic address space}}
}
