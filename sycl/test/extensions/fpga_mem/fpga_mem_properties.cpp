// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

// Test memory properties can be applied to fpga_mem

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

namespace oneapi = sycl::ext::oneapi::experimental; // for property queries
namespace intel = sycl::ext::intel::experimental;   // for fpga_mem

static intel::fpga_mem<int, decltype(oneapi::properties(intel::num_banks<888>))>
    mem_num;
static intel::fpga_mem<int> mem_empty;
static intel::fpga_mem<int, decltype(oneapi::properties(intel::clock_2x_true))>
    mem_bool;
static intel::fpga_mem<int, decltype(oneapi::properties(intel::resource_mlab))>
    mem_enum;
static intel::fpga_mem<int, decltype(oneapi::properties(
                                intel::ram_stitching_min_ram,
                                intel::stride_size<777>,
                                intel::bi_directional_ports_true))>
    mem_multi;

// Checks is_property_key_of and oneapi::is_property_value_of for T.
template <typename T> void checkIsPropertyOf() {
  static_assert(oneapi::is_property_key_of<intel::resource_key, T>::value);
  static_assert(oneapi::is_property_key_of<intel::num_banks_key, T>::value);
  static_assert(oneapi::is_property_key_of<intel::stride_size_key, T>::value);
  static_assert(oneapi::is_property_key_of<intel::word_size_key, T>::value);
  static_assert(
      oneapi::is_property_key_of<intel::bi_directional_ports_key, T>::value);
  static_assert(oneapi::is_property_key_of<intel::clock_2x_key, T>::value);
  static_assert(oneapi::is_property_key_of<intel::ram_stitching_key, T>::value);
  static_assert(
      oneapi::is_property_key_of<intel::max_private_copies_key, T>::value);
  static_assert(
      oneapi::is_property_key_of<intel::num_replicates_key, T>::value);
}

int main() {
  // Are all keys usable
  static_assert(oneapi::is_property_key<intel::resource_key>::value);
  static_assert(oneapi::is_property_key<intel::num_banks_key>::value);
  static_assert(oneapi::is_property_key<intel::stride_size_key>::value);
  static_assert(oneapi::is_property_key<intel::word_size_key>::value);
  static_assert(
      oneapi::is_property_key<intel::bi_directional_ports_key>::value);
  static_assert(oneapi::is_property_key<intel::clock_2x_key>::value);
  static_assert(oneapi::is_property_key<intel::ram_stitching_key>::value);
  static_assert(oneapi::is_property_key<intel::max_private_copies_key>::value);
  static_assert(oneapi::is_property_key<intel::num_replicates_key>::value);

  // Are all common values usable
  static_assert(
      oneapi::is_property_value<decltype(intel::resource_mlab)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::resource_block_ram)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::num_banks<8>)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::stride_size<8>)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::word_size<32>)>::value);
  static_assert(oneapi::is_property_value<
                decltype(intel::bi_directional_ports_false)>::value);
  static_assert(oneapi::is_property_value<
                decltype(intel::bi_directional_ports_true)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::clock_2x_false)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::clock_2x_true)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::ram_stitching_min_ram)>::value);
  static_assert(oneapi::is_property_value<
                decltype(intel::ram_stitching_max_fmax)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::max_private_copies<8>)>::value);
  static_assert(
      oneapi::is_property_value<decltype(intel::num_replicates<8>)>::value);

  // Check that only the property that are expected are associated with obj
  checkIsPropertyOf<decltype(mem_num)>();
  static_assert(mem_num.has_property<intel::num_banks_key>());
  static_assert(!mem_num.has_property<intel::word_size_key>());
  static_assert(mem_num.get_property<intel::num_banks_key>().value == 888);

  checkIsPropertyOf<decltype(mem_empty)>();
  static_assert(!mem_empty.has_property<intel::word_size_key>());

  checkIsPropertyOf<decltype(mem_bool)>();
  static_assert(mem_bool.has_property<intel::clock_2x_key>());
  static_assert(!mem_bool.has_property<intel::word_size_key>());
  static_assert(mem_bool.get_property<intel::clock_2x_key>().value == true);

  checkIsPropertyOf<decltype(mem_enum)>();
  static_assert(mem_enum.has_property<intel::resource_key>());
  static_assert(!mem_enum.has_property<intel::word_size_key>());
  static_assert(mem_enum.get_property<intel::resource_key>().value ==
                intel::resource_enum::mlab);

  checkIsPropertyOf<decltype(mem_multi)>();
  static_assert(mem_multi.has_property<intel::ram_stitching_key>());
  static_assert(mem_multi.has_property<intel::stride_size_key>());
  static_assert(mem_multi.has_property<intel::bi_directional_ports_key>());
  static_assert(!mem_multi.has_property<intel::word_size_key>());
  static_assert(mem_multi.get_property<intel::ram_stitching_key>().value ==
                intel::ram_stitching_enum::min_ram);
  static_assert(mem_multi.get_property<intel::stride_size_key>().value == 777);
  static_assert(
      mem_multi.get_property<intel::bi_directional_ports_key>().value == true);

  return 0;
}
