// Use update_test.sh to (re-)generate the checks
// REQUIRES: linux
// RUN: bash %S/deps_known.sh sycl/accessor.hpp | FileCheck %s

// CHECK-LABEL: Dependencies for <sycl/accessor.hpp>:
// CHECK-NEXT: /dev/null: /dev/null
// CHECK-NEXT: accessor.hpp
// CHECK-NEXT: access/access.hpp
// CHECK-NEXT: detail/defines_elementary.hpp
// CHECK-NEXT: detail/accessor_iterator.hpp
// CHECK-NEXT: detail/fwd/accessor.hpp
// CHECK-NEXT: detail/defines.hpp
// CHECK-NEXT: id.hpp
// CHECK-NEXT: detail/array.hpp
// CHECK-NEXT: range.hpp
// CHECK-NEXT: detail/code_location.hpp
// CHECK-NEXT: detail/export.hpp
// CHECK-NEXT: detail/common.hpp
// CHECK-NEXT: detail/assert.hpp
// CHECK-NEXT: __spirv/spirv_vars.hpp
// CHECK-NEXT: detail/nd_loop.hpp
// CHECK-NEXT: detail/fwd/buffer.hpp
// CHECK-NEXT: detail/generic_type_traits.hpp
// CHECK-NEXT: aliases.hpp
// CHECK-NEXT: bit_cast.hpp
// CHECK-NEXT: detail/fwd/half.hpp
// CHECK-NEXT: detail/type_traits.hpp
// CHECK-NEXT: detail/type_traits/bool_traits.hpp
// CHECK-NEXT: detail/type_traits/vec_marray_traits.hpp
// CHECK-NEXT: detail/fwd/multi_ptr.hpp
// CHECK-NEXT: detail/type_traits/integer_traits.hpp
// CHECK-NEXT: detail/handler_proxy.hpp
// CHECK-NEXT: detail/loop.hpp
// CHECK-NEXT: detail/owner_less_base.hpp
// CHECK-NEXT: detail/impl_utils.hpp
// CHECK-NEXT: ext/oneapi/weak_object_base.hpp
// CHECK-NEXT: detail/property_helper.hpp
// CHECK-NEXT: detail/property_list_base.hpp
// CHECK-NEXT: exception.hpp
// CHECK-NEXT: detail/string.hpp
// CHECK-NEXT: ext/oneapi/accessor_property_list.hpp
// CHECK-NEXT: property_list.hpp
// CHECK-NEXT: properties/property_traits.hpp
// CHECK-NEXT: multi_ptr.hpp
// CHECK-NEXT: detail/address_space_cast.hpp
// CHECK-NEXT: __spirv/spirv_types.hpp
// CHECK-NEXT: pointers.hpp
// CHECK-NEXT: properties/accessor_properties.hpp
// CHECK-NEXT: properties/runtime_accessor_properties.def
// CHECK-EMPTY:
