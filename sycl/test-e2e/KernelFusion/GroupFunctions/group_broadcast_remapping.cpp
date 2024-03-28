// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out

// Test fusion works with group_broadcast and remapping.

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/properties/all_properties.hpp>

#include "../helpers.hpp"

using namespace sycl;

class FillKernel;
class Kernel0;
class Kernel1;

int main() {
  constexpr size_t dataSize = 512;
  constexpr size_t localSize = 16;
  std::array<int, dataSize / localSize> in;
  std::array<int, dataSize> out0;
  std::array<int, dataSize / 2> out1;

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  {
    buffer<int> buff_in{in};
    buffer<int> buff_out0{out0};
    buffer<int> buff_out1{out1};

    ext::codeplay::experimental::fusion_wrapper fw{q};

    q.submit([&](handler &cgh) {
      accessor in(buff_in, cgh, write_only, no_init);
      cgh.parallel_for<FillKernel>(nd_range<1>{{dataSize}, {localSize}},
                                   [=](nd_item<1> i) {
                                     if (i.get_local_id() == 0) {
                                       auto j = i.get_group(0);
                                       in[j] = static_cast<int>(j);
                                     }
                                   });
    });

    fw.start_fusion();

    q.submit([&](handler &cgh) {
      accessor in(buff_in, cgh, read_only);
      accessor out(buff_out0, cgh, write_only, no_init);
      cgh.parallel_for<Kernel0>(
          nd_range<1>{{dataSize}, {localSize}}, [=](nd_item<1> i) {
            auto group = i.get_group();
            out[i.get_global_id()] = group_broadcast(
                group, i.get_local_id() == 1 ? in[group.get_group_id(0)] : -1,
                1);
          });
    });

    q.submit([&](handler &cgh) {
      accessor in(buff_in, cgh, read_only);
      accessor out(buff_out1, cgh, write_only, no_init);
      cgh.parallel_for<Kernel1>(
          nd_range<1>{{dataSize / 2}, {localSize}}, [=](nd_item<1> i) {
            auto group = i.get_group();
            out[i.get_global_id()] = group_broadcast(
                group, i.get_local_id() == 1 ? in[group.get_group_id(0)] : -1,
                1);
          });
    });

    complete_fusion_with_check(fw);
  }

  // Check the results
  int i = 0;
  for (int end = dataSize / 2; i < end; ++i) {
    int group_id = i / static_cast<int>(localSize);
    assert(out0[i] == group_id && "Computation error");
    assert(out1[i] == group_id && "Computation error");
  }

  for (int end = dataSize; i < end; ++i) {
    int group_id = i / static_cast<int>(localSize);
    assert(out0[i] == group_id && "Computation error");
  }

  return 0;
}
