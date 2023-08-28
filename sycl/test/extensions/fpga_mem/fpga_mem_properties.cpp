// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/properties.hpp>

using namespace sycl::ext::oneapi::experimental; // for property queries

static sycl::ext::intel::experimental::fpga_mem<int, decltype(properties(num_banks<888>))> mem1;

// Checks is_property_key_of and is_property_value_of for T.
template <typename T> void checkIsPropertyOf() {
  static_assert(is_property_key_of<num_banks_key, T>::value);
}

int main() {
  // Are all keys usable
  static_assert(is_property_key<resource_key>::value);
  static_assert(is_property_key<num_banks_key>::value);
  static_assert(is_property_key<stride_size_key>::value);
  static_assert(is_property_key<word_size_key>::value);
  static_assert(is_property_key<bi_directional_ports_key>::value);
  static_assert(is_property_key<clock_2x_key>::value);
  static_assert(is_property_key<ram_stitching_key>::value);
  static_assert(is_property_key<max_private_copies_key>::value);
  static_assert(is_property_key<num_replicates_key>::value);

  // Are all common values usable
  static_assert(is_property_value<decltype(resource_mlab)>::value);
  static_assert(is_property_value<decltype(resource_block_ram)>::value);
  // FIXME // static_assert(is_property_value<decltype(resource_block_any)>::value);
  static_assert(is_property_value<decltype(num_banks<8>)>::value);
  static_assert(is_property_value<decltype(stride_size<8>)>::value);
  static_assert(is_property_value<decltype(word_size<32>)>::value);
  static_assert(is_property_value<decltype(bi_directional_ports_false)>::value);
  static_assert(is_property_value<decltype(bi_directional_ports_true)>::value);
  static_assert(is_property_value<decltype(clock_2x_false)>::value);
  static_assert(is_property_value<decltype(clock_2x_true)>::value);
  static_assert(is_property_value<decltype(ram_stitching_min_ram)>::value);
  static_assert(is_property_value<decltype(ram_stitching_max_fmax)>::value);
  static_assert(is_property_value<decltype(max_private_copies<8>)>::value);
  static_assert(is_property_value<decltype(num_replicates<8>)>::value);


  checkIsPropertyOf<decltype(mem1)>(); //Fail
  // static_assert(mem1.has_property<num_banks>());
  // static_assert(!mem1.has_property<host_access_key>());
  // static_assert(!mem1.has_property<init_mode_key>());
  // static_assert(!mem1.has_property<implement_in_csr_key>());
  // static_assert(mem1.get_property<num_banks>() ==
  //               num_banks<888>);

  return 0;
}
