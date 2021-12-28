// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl-device-only %s -o - | FileCheck %s

#include <CL/sycl.hpp>

using namespace sycl;

template <typename T, template <typename> class O, typename G>
void group_operations(G g) {
  T res;
  res = reduce_over_group(g, T{1}, O<T>{});
  res = inclusive_scan_over_group(g, T{1}, O<T>{});
  res = exclusive_scan_over_group(g, T{1}, O<T>{});
}

template <typename T, typename G> void binary_operations(G g) {
  if constexpr (!detail::is_floating_point<T>::value) {
    group_operations<T, bit_and>(g);
    group_operations<T, bit_or>(g);
    group_operations<T, bit_xor>(g);
  }
  group_operations<T, maximum>(g);
  group_operations<T, minimum>(g);
  group_operations<T, multiplies>(g);
  group_operations<T, plus>(g);
}

template <typename T, typename G> void signed_unsigned(G g) {
  binary_operations<T>(g);
  group_broadcast(g, T{1});
  group_broadcast(g, T{1}, typename G::linear_id_type{1});
  group_broadcast(g, T{1}, typename G::id_type{2});

  if constexpr (!detail::is_floating_point<T>::value) {
    binary_operations<detail::make_unsigned_t<T>>(g);
    group_broadcast(g, detail::make_unsigned_t<T>{1});
    group_broadcast(g, detail::make_unsigned_t<T>{1},
                    typename G::linear_id_type{1});
    group_broadcast(g, detail::make_unsigned_t<T>{1}, typename G::id_type{2});
  }
}

template <typename G> void test(G g) {
  signed_unsigned<int8_t>(g);
  signed_unsigned<int16_t>(g);
  signed_unsigned<int32_t>(g);
  signed_unsigned<int64_t>(g);
  signed_unsigned<half>(g);
  signed_unsigned<float>(g);
  signed_unsigned<double>(g);
}

SYCL_EXTERNAL void test_group(group<> g) { test(g); }

//
// Broadcast
//

// int8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 2)

// uint8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 2)

// int16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 2)

// uint16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 2)

// int32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastji{{m|y}}(i32 2, i32 1, i64 2)

// uint32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 1, i64 2)

// int64_t (Linux: long, Windows: long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{l|x}}{{m|y}}(i32 2, i64 1, i64 0)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{l|x}}{{m|y}}(i32 2, i64 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{l|x}}{{m|y}}(i32 2, i64 1, i64 2)

// uint64_t (Linux: long, Windows: long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{m|y}}{{m|y}}(i32 2, i64 1, i64 0)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{m|y}}{{m|y}}(i32 2, i64 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{m|y}}{{m|y}}(i32 2, i64 1, i64 2)

// half (15360 = 0xH3C00 = 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 15360, i64 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 15360, i64 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjj{{m|y}}(i32 2, i32 15360, i64 2)

// float
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupBroadcastjf{{m|y}}(i32 2, float 1.000000e+00, i64 0)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupBroadcastjf{{m|y}}(i32 2, float 1.000000e+00, i64 1)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupBroadcastjf{{m|y}}(i32 2, float 1.000000e+00, i64 2)

// double
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupBroadcastjd{{m|y}}(i32 2, double 1.000000e+00, i64 0)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupBroadcastjd{{m|y}}(i32 2, double 1.000000e+00, i64 1)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupBroadcastjd{{m|y}}(i32 2, double 1.000000e+00, i64 2)

//
// Binary operations
//

// int8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 2, i32 1)

// uint8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 2, i32 1)

// int16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 2, i32 1)

// uint16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 2, i32 1)

// int32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 2, i32 2, i32 1)

// uint32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 2, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 2, i32 2, i32 1)

// int64_t (Linux: long, Windows: long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{l|x}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{l|x}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{l|x}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{l|x}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{l|x}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{l|x}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{l|x}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{l|x}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{l|x}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMaxjj{{l|x}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMaxjj{{l|x}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMaxjj{{l|x}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMinjj{{l|x}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMinjj{{l|x}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMinjj{{l|x}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{l|x}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{l|x}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{l|x}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{l|x}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{l|x}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{l|x}}(i32 2, i32 2, i64 1)

// uint64_t (Linux: unsigned long, Windows: unsigned long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{m|y}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{m|y}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{m|y}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{m|y}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{m|y}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{m|y}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMaxjj{{m|y}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMaxjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMaxjj{{m|y}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMinjj{{m|y}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMinjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMinjj{{m|y}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{m|y}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{m|y}}(i32 2, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{m|y}}(i32 2, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{m|y}}(i32 2, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{m|y}}(i32 2, i32 2, i64 1)

// half (0xH3C00 = 1)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMaxjjDF16_(i32 2, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMaxjjDF16_(i32 2, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMaxjjDF16_(i32 2, i32 2, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMinjjDF16_(i32 2, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMinjjDF16_(i32 2, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMinjjDF16_(i32 2, i32 2, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupNonUniformFMuljjDF16_(i32 2, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupNonUniformFMuljjDF16_(i32 2, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupNonUniformFMuljjDF16_(i32 2, i32 2, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFAddjjDF16_(i32 2, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFAddjjDF16_(i32 2, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFAddjjDF16_(i32 2, i32 2, half 0xH3C00)

// float
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMaxjjf(i32 2, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMaxjjf(i32 2, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMaxjjf(i32 2, i32 2, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMinjjf(i32 2, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMinjjf(i32 2, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMinjjf(i32 2, i32 2, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupNonUniformFMuljjf(i32 2, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupNonUniformFMuljjf(i32 2, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupNonUniformFMuljjf(i32 2, i32 2, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFAddjjf(i32 2, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFAddjjf(i32 2, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFAddjjf(i32 2, i32 2, float 1.000000e+00)

// double
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMaxjjd(i32 2, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMaxjjd(i32 2, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMaxjjd(i32 2, i32 2, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMinjjd(i32 2, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMinjjd(i32 2, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMinjjd(i32 2, i32 2, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupNonUniformFMuljjd(i32 2, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupNonUniformFMuljjd(i32 2, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupNonUniformFMuljjd(i32 2, i32 2, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFAddjjd(i32 2, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFAddjjd(i32 2, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFAddjjd(i32 2, i32 2, double 1.000000e+00)

SYCL_EXTERNAL void test_sub_group(sub_group g) { test(g); }

//
// Broadcast
//

// int8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 2)

// uint8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 2)

// int16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 2)

// uint16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 2)

// int32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjij(i32 3, i32 1, i32 2)

// uint32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 1, i32 2)

// int64_t (Linux: long, Windows: long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{l|x}}j(i32 3, i64 1, i32 0)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{l|x}}j(i32 3, i64 1, i32 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{l|x}}j(i32 3, i64 1, i32 2)

// uint64_t (Linux: long, Windows: long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{m|y}}j(i32 3, i64 1, i32 0)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{m|y}}j(i32 3, i64 1, i32 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupBroadcastj{{m|y}}j(i32 3, i64 1, i32 2)

// half (15360 = 0xH3C00 = 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 15360, i32 0)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 15360, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupBroadcastjjj(i32 3, i32 15360, i32 2)

// float
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupBroadcastjfj(i32 3, float 1.000000e+00, i32 0)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupBroadcastjfj(i32 3, float 1.000000e+00, i32 1)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupBroadcastjfj(i32 3, float 1.000000e+00, i32 2)

// double
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupBroadcastjdj(i32 3, double 1.000000e+00, i32 0)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupBroadcastjdj(i32 3, double 1.000000e+00, i32 1)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupBroadcastjdj(i32 3, double 1.000000e+00, i32 2)

//
// Binary operations
//

// int8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 2, i32 1)

// uint8_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 2, i32 1)

// int16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 2, i32 1)

// uint16_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 2, i32 1)

// int32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMaxjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupSMinjji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljji(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjji(i32 3, i32 2, i32 1)

// uint32_t
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMaxjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupUMinjjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupNonUniformIMuljjj(i32 3, i32 2, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 0, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 1, i32 1)
// CHECK: call spir_func i32 @_Z[[#]]__spirv_GroupIAddjjj(i32 3, i32 2, i32 1)

// int64_t (Linux: long, Windows: long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{l|x}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{l|x}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{l|x}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{l|x}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{l|x}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{l|x}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{l|x}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{l|x}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{l|x}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMaxjj{{l|x}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMaxjj{{l|x}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMaxjj{{l|x}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMinjj{{l|x}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMinjj{{l|x}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupSMinjj{{l|x}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{l|x}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{l|x}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{l|x}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{l|x}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{l|x}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{l|x}}(i32 3, i32 2, i64 1)

// uint64_t (Linux: unsigned long, Windows: unsigned long long)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{m|y}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{m|y}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseAndjj{{m|y}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{m|y}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{m|y}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseOrjj{{m|y}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{m|y}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{m|y}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformBitwiseXorjj{{m|y}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMaxjj{{m|y}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMaxjj{{m|y}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMaxjj{{m|y}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMinjj{{m|y}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMinjj{{m|y}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupUMinjj{{m|y}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{m|y}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{m|y}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupNonUniformIMuljj{{m|y}}(i32 3, i32 2, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{m|y}}(i32 3, i32 0, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{m|y}}(i32 3, i32 1, i64 1)
// CHECK: call spir_func i64 @_Z[[#]]__spirv_GroupIAddjj{{m|y}}(i32 3, i32 2, i64 1)

// half (0xH3C00 = 1)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMaxjjDF16_(i32 3, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMaxjjDF16_(i32 3, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMaxjjDF16_(i32 3, i32 2, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMinjjDF16_(i32 3, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMinjjDF16_(i32 3, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFMinjjDF16_(i32 3, i32 2, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupNonUniformFMuljjDF16_(i32 3, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupNonUniformFMuljjDF16_(i32 3, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupNonUniformFMuljjDF16_(i32 3, i32 2, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFAddjjDF16_(i32 3, i32 0, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFAddjjDF16_(i32 3, i32 1, half 0xH3C00)
// CHECK: call spir_func half @_Z[[#]]__spirv_GroupFAddjjDF16_(i32 3, i32 2, half 0xH3C00)

// float
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMaxjjf(i32 3, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMaxjjf(i32 3, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMaxjjf(i32 3, i32 2, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMinjjf(i32 3, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMinjjf(i32 3, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFMinjjf(i32 3, i32 2, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupNonUniformFMuljjf(i32 3, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupNonUniformFMuljjf(i32 3, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupNonUniformFMuljjf(i32 3, i32 2, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFAddjjf(i32 3, i32 0, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFAddjjf(i32 3, i32 1, float 1.000000e+00)
// CHECK: call spir_func float @_Z[[#]]__spirv_GroupFAddjjf(i32 3, i32 2, float 1.000000e+00)

// double
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMaxjjd(i32 3, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMaxjjd(i32 3, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMaxjjd(i32 3, i32 2, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMinjjd(i32 3, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMinjjd(i32 3, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFMinjjd(i32 3, i32 2, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupNonUniformFMuljjd(i32 3, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupNonUniformFMuljjd(i32 3, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupNonUniformFMuljjd(i32 3, i32 2, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFAddjjd(i32 3, i32 0, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFAddjjd(i32 3, i32 1, double 1.000000e+00)
// CHECK: call spir_func double @_Z[[#]]__spirv_GroupFAddjjd(i32 3, i32 2, double 1.000000e+00)
