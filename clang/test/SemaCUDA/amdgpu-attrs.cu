// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
#include "Inputs/cuda.h"


__attribute__((amdgpu_flat_work_group_size(32, 64)))
__global__ void flat_work_group_size_32_64() {}

__attribute__((amdgpu_waves_per_eu(2)))
__global__ void waves_per_eu_2() {}

__attribute__((amdgpu_waves_per_eu(2, 4)))
__global__ void waves_per_eu_2_4() {}

__attribute__((amdgpu_num_sgpr(32)))
__global__ void num_sgpr_32() {}

__attribute__((amdgpu_num_vgpr(64)))
__global__ void num_vgpr_64() {}


__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_num_sgpr_32() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_num_vgpr_64() {}

__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32)))
__global__ void waves_per_eu_2_num_sgpr_32() {}

__attribute__((amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64)))
__global__ void waves_per_eu_2_num_vgpr_64() {}

__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32)))
__global__ void waves_per_eu_2_4_num_sgpr_32() {}

__attribute__((amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64)))
__global__ void waves_per_eu_2_4_num_vgpr_64() {}

__attribute__((amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void num_sgpr_32_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_num_sgpr_32_num_vgpr_64() {}

__attribute__((amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32_num_vgpr_64() {}

__attribute__((amdgpu_max_num_work_groups(32, 1, 1)))
__global__ void max_num_work_groups_32_1_1() {}

__attribute__((amdgpu_max_num_work_groups(32, 1, 1), amdgpu_flat_work_group_size(32, 64)))
__global__ void max_num_work_groups_32_1_1_flat_work_group_size_32_64() {}

__attribute__((amdgpu_max_num_work_groups(32, 1, 1), amdgpu_flat_work_group_size(32, 64), amdgpu_waves_per_eu(2, 4), amdgpu_num_sgpr(32), amdgpu_num_vgpr(64)))
__global__ void max_num_work_groups_32_1_1_flat_work_group_size_32_64_waves_per_eu_2_4_num_sgpr_32_num_vgpr_64() {}


// expected-error@+2{{attribute 'reqd_work_group_size' can only be applied to an OpenCL kernel function}}
__attribute__((reqd_work_group_size(32, 64, 64)))
__global__ void reqd_work_group_size_32_64_64() {}

// expected-error@+2{{attribute 'work_group_size_hint' can only be applied to an OpenCL kernel function}}
__attribute__((work_group_size_hint(2, 2, 2)))
__global__ void work_group_size_hint_2_2_2() {}

// expected-error@+2{{attribute 'vec_type_hint' can only be applied to an OpenCL kernel function}}
__attribute__((vec_type_hint(int)))
__global__ void vec_type_hint_int() {}

// expected-warning@+1{{'intel_reqd_sub_group_size' attribute ignored}}
__attribute__((intel_reqd_sub_group_size(64)))
__global__ void intel_reqd_sub_group_size_64() {}

// expected-error@+1{{'amdgpu_flat_work_group_size' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_flat_work_group_size("32", 64)))
__global__ void non_int_min_flat_work_group_size_32_64() {}
// expected-error@+1{{'amdgpu_flat_work_group_size' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_flat_work_group_size(32, "64")))
__global__ void non_int_max_flat_work_group_size_32_64() {}

int nc_min = 32, nc_max = 64;
// expected-error@+1{{'amdgpu_flat_work_group_size' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_flat_work_group_size(nc_min, 64)))
__global__ void non_cint_min_flat_work_group_size_32_64() {}
// expected-error@+1{{'amdgpu_flat_work_group_size' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_flat_work_group_size(32, nc_max)))
__global__ void non_cint_max_flat_work_group_size_32_64() {}

const int c_min = 16, c_max = 32;
__attribute__((amdgpu_flat_work_group_size(c_min * 2, 64)))
__global__ void cint_min_flat_work_group_size_32_64() {}
__attribute__((amdgpu_flat_work_group_size(32, c_max * 2)))
__global__ void cint_max_flat_work_group_size_32_64() {}

// expected-error@+3{{'T' does not refer to a value}}
// expected-note@+1{{declared here}}
template<typename T>
__attribute__((amdgpu_flat_work_group_size(T, 64)))
__global__ void template_class_min_flat_work_group_size_32_64() {}
// expected-error@+3{{'T' does not refer to a value}}
// expected-note@+1{{declared here}}
template<typename T>
__attribute__((amdgpu_flat_work_group_size(32, T)))
__global__ void template_class_max_flat_work_group_size_32_64() {}

template<unsigned a, unsigned b>
__attribute__((amdgpu_flat_work_group_size(a, b)))
__global__ void template_flat_work_group_size_32_64() {}
template __global__ void template_flat_work_group_size_32_64<32, 64>();

template<unsigned a, unsigned b, unsigned c>
__attribute__((amdgpu_flat_work_group_size(a + b, b + c)))
__global__ void template_complex_flat_work_group_size_32_64() {}
template __global__ void template_complex_flat_work_group_size_32_64<16, 16, 48>();

unsigned ipow2(unsigned n) { return n == 0 ? 1 : 2 * ipow2(n - 1); }
constexpr unsigned ce_ipow2(unsigned n) { return n == 0 ? 1 : 2 * ce_ipow2(n - 1); }

__attribute__((amdgpu_flat_work_group_size(ce_ipow2(5), ce_ipow2(6))))
__global__ void cexpr_flat_work_group_size_32_64() {}
// expected-error@+1{{'amdgpu_flat_work_group_size' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_flat_work_group_size(ipow2(5), 64)))
__global__ void non_cexpr_min_flat_work_group_size_32_64() {}
// expected-error@+1{{'amdgpu_flat_work_group_size' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_flat_work_group_size(32, ipow2(6))))
__global__ void non_cexpr_max_flat_work_group_size_32_64() {}

// expected-error@+1{{'amdgpu_waves_per_eu' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_waves_per_eu("2")))
__global__ void non_int_min_waves_per_eu_2() {}
// expected-error@+1{{'amdgpu_waves_per_eu' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_waves_per_eu(2, "4")))
__global__ void non_int_max_waves_per_eu_2_4() {}

// expected-error@+1{{'amdgpu_waves_per_eu' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_waves_per_eu(nc_min)))
__global__ void non_cint_min_waves_per_eu_2() {}
// expected-error@+1{{'amdgpu_waves_per_eu' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_waves_per_eu(2, nc_max)))
__global__ void non_cint_min_waves_per_eu_2_4() {}

__attribute__((amdgpu_waves_per_eu(c_min / 8)))
__global__ void cint_min_waves_per_eu_2() {}
__attribute__((amdgpu_waves_per_eu(c_min / 8, c_max / 8)))
__global__ void cint_min_waves_per_eu_2_4() {}

// expected-error@+3{{'T' does not refer to a value}}
// expected-note@+1{{declared here}}
template<typename T>
__attribute__((amdgpu_waves_per_eu(T)))
__global__ void cint_min_waves_per_eu_2() {}
// expected-error@+3{{'T' does not refer to a value}}
// expected-note@+1{{declared here}}
template<typename T>
__attribute__((amdgpu_waves_per_eu(2, T)))
__global__ void cint_min_waves_per_eu_2_4() {}

template<unsigned a>
__attribute__((amdgpu_waves_per_eu(a)))
__global__ void template_waves_per_eu_2() {}
template __global__ void template_waves_per_eu_2<2>();

template<unsigned a, unsigned b>
__attribute__((amdgpu_waves_per_eu(a, b)))
__global__ void template_waves_per_eu_2_4() {}
template __global__ void template_waves_per_eu_2_4<2, 4>();

template<unsigned a, unsigned b, unsigned c>
__attribute__((amdgpu_waves_per_eu(a + b, c - b)))
__global__ void template_complex_waves_per_eu_2_4() {}
template __global__ void template_complex_waves_per_eu_2_4<1, 1, 5>();

// expected-error@+2{{expression contains unexpanded parameter pack 'Args'}}
template<unsigned... Args>
__attribute__((amdgpu_waves_per_eu(Args)))
__global__ void template_waves_per_eu_2() {}
template __global__ void template_waves_per_eu_2<2, 4>();

__attribute__((amdgpu_waves_per_eu(ce_ipow2(1))))
__global__ void cexpr_waves_per_eu_2() {}
__attribute__((amdgpu_waves_per_eu(ce_ipow2(1), ce_ipow2(2))))
__global__ void cexpr_waves_per_eu_2_4() {}
// expected-error@+1{{'amdgpu_waves_per_eu' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_waves_per_eu(ipow2(1))))
__global__ void non_cexpr_waves_per_eu_2() {}
// expected-error@+1{{'amdgpu_waves_per_eu' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_waves_per_eu(2, ipow2(2))))
__global__ void non_cexpr_waves_per_eu_2_4() {}

__attribute__((amdgpu_max_num_work_groups(32)))
__global__ void max_num_work_groups_32() {}

__attribute__((amdgpu_max_num_work_groups(32, 1)))
__global__ void max_num_work_groups_32_1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute takes no more than 3 arguments}}
__attribute__((amdgpu_max_num_work_groups(32, 1, 1, 1)))
__global__ void max_num_work_groups_32_1_1_1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute takes at least 1 argument}}
__attribute__((amdgpu_max_num_work_groups()))
__global__ void max_num_work_groups_no_arg() {}

// expected-error@+1{{expected expression}}
__attribute__((amdgpu_max_num_work_groups(,1,1)))
__global__ void max_num_work_groups_empty_1_1() {}

// expected-error@+1{{expected expression}}
__attribute__((amdgpu_max_num_work_groups(32,,1)))
__global__ void max_num_work_groups_32_empty_1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_max_num_work_groups(ipow2(5), 1, 1)))
__global__ void max_num_work_groups_32_1_1_non_int_arg0() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_max_num_work_groups(32, "1", 1)))
__global__ void max_num_work_groups_32_1_1_non_int_arg1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires a non-negative integral compile time constant expression}}
__attribute__((amdgpu_max_num_work_groups(-32, 1, 1)))
__global__ void max_num_work_groups_32_1_1_neg_int_arg0() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires a non-negative integral compile time constant expression}}
__attribute__((amdgpu_max_num_work_groups(32, -1, 1)))
__global__ void max_num_work_groups_32_1_1_neg_int_arg1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires a non-negative integral compile time constant expression}}
__attribute__((amdgpu_max_num_work_groups(32, 1, -1)))
__global__ void max_num_work_groups_32_1_1_neg_int_arg2() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute must be greater than 0}}
__attribute__((amdgpu_max_num_work_groups(0, 1, 1)))
__global__ void max_num_work_groups_0_1_1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute must be greater than 0}}
__attribute__((amdgpu_max_num_work_groups(32, 0, 1)))
__global__ void max_num_work_groups_32_0_1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute must be greater than 0}}
__attribute__((amdgpu_max_num_work_groups(32, 1, 0)))
__global__ void max_num_work_groups_32_1_0() {}

__attribute__((amdgpu_max_num_work_groups(4294967295)))
__global__ void max_num_work_groups_max_unsigned_int() {}

// expected-error@+1{{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_max_num_work_groups(4294967296)))
__global__ void max_num_work_groups_max_unsigned_int_plus1() {}

// expected-error@+1{{integer constant expression evaluates to value 10000000000 that cannot be represented in a 32-bit unsigned integer type}}
__attribute__((amdgpu_max_num_work_groups(10000000000)))
__global__ void max_num_work_groups_too_large() {}

int num_wg_x = 32;
int num_wg_y = 1;
int num_wg_z = 1;
// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires parameter 0 to be an integer constant}}
__attribute__((amdgpu_max_num_work_groups(num_wg_x, 1, 1)))
__global__ void max_num_work_groups_32_1_1_non_const_arg0() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires parameter 1 to be an integer constant}}
__attribute__((amdgpu_max_num_work_groups(32, num_wg_y, 1)))
__global__ void max_num_work_groups_32_1_1_non_const_arg1() {}

// expected-error@+1{{'amdgpu_max_num_work_groups' attribute requires parameter 2 to be an integer constant}}
__attribute__((amdgpu_max_num_work_groups(32, 1, num_wg_z)))
__global__ void max_num_work_groups_32_1_1_non_const_arg2() {}

const int c_num_wg_x = 32;
__attribute__((amdgpu_max_num_work_groups(c_num_wg_x, 1, 1)))
__global__ void max_num_work_groups_32_1_1_const_arg0() {}

template<unsigned a>
__attribute__((amdgpu_max_num_work_groups(a, 1, 1)))
__global__ void template_a_1_1_max_num_work_groups() {}
template __global__ void template_a_1_1_max_num_work_groups<32>();

template<unsigned a>
__attribute__((amdgpu_max_num_work_groups(32, a, 1)))
__global__ void template_32_a_1_max_num_work_groups() {}
template __global__ void template_32_a_1_max_num_work_groups<1>();

template<unsigned a>
__attribute__((amdgpu_max_num_work_groups(32, 1, a)))
__global__ void template_32_1_a_max_num_work_groups() {}
template __global__ void template_32_1_a_max_num_work_groups<1>();

// expected-error@+3{{'amdgpu_max_num_work_groups' attribute must be greater than 0}}
// expected-note@+4{{in instantiation of}}
template<unsigned b>
__attribute__((amdgpu_max_num_work_groups(b, 1, 1)))
__global__ void template_b_1_1_max_num_work_groups() {}
template __global__ void template_b_1_1_max_num_work_groups<0>();

// expected-error@+3{{'amdgpu_max_num_work_groups' attribute must be greater than 0}}
// expected-note@+4{{in instantiation of}}
template<unsigned b>
__attribute__((amdgpu_max_num_work_groups(32, b, 1)))
__global__ void template_32_b_1_max_num_work_groups() {}
template __global__ void template_32_b_1_max_num_work_groups<0>();

// expected-error@+3{{'amdgpu_max_num_work_groups' attribute must be greater than 0}}
// expected-note@+4{{in instantiation of}}
template<unsigned b>
__attribute__((amdgpu_max_num_work_groups(32, 1, b)))
__global__ void template_32_1_b_max_num_work_groups() {}
template __global__ void template_32_1_b_max_num_work_groups<0>();


