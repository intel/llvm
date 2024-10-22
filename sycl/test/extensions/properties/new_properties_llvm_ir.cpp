// RUN: %clangxx -fsycl -fsycl-device-only -O0 -emit-llvm -S %s -o - | FileCheck %s

// CHECK: @fg_int = linkonce_odr dso_local addrspace(1) global %struct.fake_device_global { i32 43 }, align 4 #[[ATTR:[0-9]*]]
// CHECK: attributes #[[ATTR]] = { "llvm-ir-prop"="42" }

#include <sycl/ext/oneapi/properties/properties.hpp>

using namespace sycl::ext::oneapi::experimental::new_properties;

template <typename property_key_t> constexpr auto generate_property_key_name() {
#if defined(__clang__) || defined(__GNUC__)
  return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
  return __FUNCSIG__;
#else
#error "Unsupported compiler"
#endif
}

template <typename property_t, typename property_key_t = property_t>
struct named_property_base
    : public detail::property_base<property_t, property_key_t> {
  static constexpr std::string_view property_name =
      generate_property_key_name<property_key_t>();
};

template <int N> struct prop : named_property_base<prop<N>, struct prop_key> {
  static constexpr const char *ir_attribute_name = "llvm-ir-prop";
  static constexpr int ir_attribute_value = N;
};

struct property_withour_ir_attribute
    : named_property_base<property_withour_ir_attribute> {};

template <typename T, typename PropListTy> struct fake_device_global;
template <typename T, typename... property_tys>
struct
    [[__sycl_detail__::global_variable_allowed,
      __sycl_detail__::add_ir_attributes_global_variable(
          property_tys::ir_attribute_name...,
          property_tys::
              ir_attribute_value...)]] fake_device_global<T,
                                                          properties<detail::properties_type_list<
                                                              property_tys...>>> {
  T value;
};

constexpr auto pl = properties{prop<42>{}, property_withour_ir_attribute{}};
using pl_t = std::remove_const_t<decltype(pl)>;

fake_device_global<int, pl_t> fg_int{43};

SYCL_EXTERNAL auto foo() { (void)fg_int; }
