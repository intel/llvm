// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -fsycl-device-only -c -fno-color-diagnostics -Xclang -fdump-record-layouts-complete %s -o %t.out | grep -Pzo "0 \| (class|struct) sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck --implicit-check-not "{{std::basic_string|std::list}}" %s
// RUN: %clangxx -std=c++17 -I %sycl_include -I %sycl_include/sycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts-complete %s -o %t.out | grep -Pzo "0 \| (class|struct) sycl::.*\n([^\n].*\n)*" | sort -z | FileCheck --implicit-check-not "{{std::basic_string|std::list}}" %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// The purpose of this test is to check that classes in sycl namespace that are
// defined in SYCL headers don't have std::string and std::list data members to
// avoid having the dual ABI issue (see
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). I.e. if
// application is built with the old ABI and such data member is crossing ABI
// boundary then it will result in issues as SYCL RT is using new ABI by
// default. All such data members can potentially cross ABI boundaries and
// that's why we need to be sure that we use only ABI-neutral data members.

// New exclusions are NOT ALLOWED to this file unless it is guaranteed that data
// member is not crossing ABI boundary. All current exclusions are listed below.

// CHECK: 0 | struct sycl::ext::oneapi::experimental::build_options
// CHECK-NEXT:         0 |   struct sycl::ext::oneapi::experimental::detail::run_time_property_key
// CHECK:              0 |   class std::vector<class std::basic_string<char> > opts
// CHECK-NEXT:         0 |     struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > > (base)
// CHECK-NEXT:         0 |       struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl _M_impl
// CHECK-NEXT:         0 |         class std::allocator<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |           class {{(std::__new_allocator|__gnu_cxx::new_allocator)}}<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |         {{(struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl_data \(base\)|pointer _M_start)}}

// CHECK: 0 | struct sycl::ext::oneapi::experimental::include_files
// CHECK-NEXT:         0 |   struct sycl::ext::oneapi::experimental::detail::run_time_property_key
// CHECK:              0 |   class std::unordered_map<class std::basic_string<char>, class std::basic_string<char> > record
// CHECK-NEXT:         0 |     class std::_Hashtable<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> > >, struct std::__detail::_Select1st, struct std::equal_to<class std::basic_string<char> >, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<true, false, true> > _M_h
// CHECK-NEXT:         0 |       struct std::__detail::_Hashtable_base<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, struct std::__detail::_Select1st, struct std::equal_to<class std::basic_string<char> >, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Hashtable_traits<true, false, true> > (base) (empty)
// CHECK-NEXT:         0 |         struct std::__detail::_Hash_code_base<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, struct std::__detail::_Select1st, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, true> (base) (empty)
// CHECK-NEXT:         0 |           struct std::__detail::_Hashtable_ebo_helper<1, struct std::hash<string> > (base) (empty)
// CHECK-NEXT:         0 |             struct std::hash<string> (base) (empty)
// CHECK-NEXT:         0 |               struct std::__hash_base<unsigned long, class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |         struct std::__detail::_Hashtable_ebo_helper<0, struct std::equal_to<class std::basic_string<char> > > (base) (empty)
// CHECK-NEXT:         0 |           struct std::equal_to<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |             struct std::binary_function<class std::basic_string<char>, class std::basic_string<char>, _Bool> (base) (empty)
// CHECK-NEXT:         0 |       struct std::__detail::_Map_base<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> > >, struct std::__detail::_Select1st, struct std::equal_to<class std::basic_string<char> >, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<true, false, true> > (base) (empty)
// CHECK-NEXT:         0 |       struct std::__detail::_Insert<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> > >, struct std::__detail::_Select1st, struct std::equal_to<class std::basic_string<char> >, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<true, false, true> > (base) (empty)
// CHECK-NEXT:         0 |         struct std::__detail::_Insert_base<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> > >, struct std::__detail::_Select1st, struct std::equal_to<class std::basic_string<char> >, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<true, false, true> > (base) (empty)
// CHECK-NEXT:         0 |       struct std::__detail::_Rehash_base<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> > >, struct std::__detail::_Select1st, struct std::equal_to<class std::basic_string<char> >, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<true, false, true> > (base) (empty)
// CHECK-NEXT:         0 |       struct std::__detail::_Equality<class std::basic_string<char>, struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> > >, struct std::__detail::_Select1st, struct std::equal_to<class std::basic_string<char> >, struct std::hash<string>, struct std::__detail::_Mod_range_hashing, struct std::__detail::_Default_ranged_hash, struct std::__detail::_Prime_rehash_policy, struct std::__detail::_Hashtable_traits<true, false, true> > (base) (empty)
// CHECK-NEXT:         0 |       struct std::__detail::_Hashtable_alloc<class std::allocator<struct std::__detail::_Hash_node<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, true> > > (base) (empty)
// CHECK-NEXT:         0 |         struct std::__detail::_Hashtable_ebo_helper<0, class std::allocator<struct std::__detail::_Hash_node<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, true> > > (base) (empty)
// CHECK-NEXT:         0 |           class std::allocator<struct std::__detail::_Hash_node<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, true> > (base) (empty)
// CHECK-NEXT:         0 |             class {{(std::__new_allocator|__gnu_cxx::new_allocator)}}<struct std::__detail::_Hash_node<struct std::pair<const class std::basic_string<char>, class std::basic_string<char> >, true> > (base) (empty)
// CHECK-NEXT:         0 |       {{(struct std::_Enable_default_constructor<true, struct std::__detail::_Hash_node_base> \(base\) \(empty\))|(__buckets_ptr _M_buckets)}}

// CHECK: 0 | struct sycl::ext::oneapi::experimental::registered_names
// CHECK-NEXT:         0 |   struct sycl::ext::oneapi::experimental::detail::run_time_property_key
// CHECK:              0 |   class std::vector<class std::basic_string<char> > names
// CHECK-NEXT:         0 |     struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > > (base)
// CHECK-NEXT:         0 |       struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl _M_impl
// CHECK-NEXT:         0 |         class std::allocator<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |           class {{(std::__new_allocator|__gnu_cxx::new_allocator)}}<class std::basic_string<char> > (base) (empty)
// CHECK-NEXT:         0 |         {{(struct std::_Vector_base<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >::_Vector_impl_data \(base\)|pointer _M_start)}}

#include <sycl/sycl.hpp>
