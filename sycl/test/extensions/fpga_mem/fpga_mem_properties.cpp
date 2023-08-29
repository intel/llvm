// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl::ext::oneapi::experimental; // for property queries
namespace intel = sycl::ext::intel::experimental; // for fpga_mem

static intel::fpga_mem<int, decltype(properties(num_banks<888>))> mem_num;
static intel::fpga_mem<int> mem_empty;
static intel::fpga_mem<int, decltype(properties(clock_2x_true))> mem_bool;
static intel::fpga_mem<int, decltype(properties(resource_mlab))> mem_enum;
static intel::fpga_mem<int, decltype(properties(ram_stitching_min_ram, stride_size<777>, bi_directional_ports_true))> mem_multi;

// Checks is_property_key_of and is_property_value_of for T.
template <typename T> void checkIsPropertyOf() {
  static_assert(is_property_key_of<resource_key, T>::value);
  static_assert(is_property_key_of<num_banks_key, T>::value);
  static_assert(is_property_key_of<stride_size_key, T>::value);
  static_assert(is_property_key_of<word_size_key, T>::value);
  static_assert(is_property_key_of<bi_directional_ports_key, T>::value);
  static_assert(is_property_key_of<clock_2x_key, T>::value);
  static_assert(is_property_key_of<ram_stitching_key, T>::value);
  static_assert(is_property_key_of<max_private_copies_key, T>::value);
  static_assert(is_property_key_of<num_replicates_key, T>::value);
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

  //Check that only the property that are expected are associated with obj
  checkIsPropertyOf<decltype(mem_num)>();
  static_assert(mem_num.has_property<num_banks_key>());
  // FIX ME // static_assert(mem_num.has_property<resource_key>());
  static_assert(!mem_num.has_property<word_size_key>());

  checkIsPropertyOf<decltype(mem_empty)>();
  // FIX ME // static_assert(mem_empty.has_property<resource_key>());
  static_assert(!mem_empty.has_property<word_size_key>());

  checkIsPropertyOf<decltype(mem_bool)>();
  // FIX ME // static_assert(mem_num.has_property<resource_key>());
  static_assert(mem_bool.has_property<clock_2x_key>());
  static_assert(!mem_bool.has_property<word_size_key>());

  checkIsPropertyOf<decltype(mem_enum)>();
  static_assert(mem_enum.has_property<resource_key>());
  static_assert(!mem_enum.has_property<word_size_key>());

  checkIsPropertyOf<decltype(mem_multi)>();
  // FIX ME // static_assert(mem_num.has_property<resource_key>());
  static_assert(mem_multi.has_property<ram_stitching_key>());
  static_assert(mem_multi.has_property<stride_size_key>());
  static_assert(mem_multi.has_property<bi_directional_ports_key>());
  static_assert(!mem_multi.has_property<word_size_key>());

  return 0;
}
