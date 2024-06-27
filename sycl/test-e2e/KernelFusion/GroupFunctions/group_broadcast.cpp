// RUN: %{build} %{embed-ir} -o %t.out
// RUN: %{run} %t.out

// Test fusion works with group_broadcast.

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/properties/all_properties.hpp>

#include "../helpers.hpp"

using namespace sycl;

class FillKernel;
class Kernel;

int main() {
  constexpr size_t dataSize = 512;
  constexpr size_t localSize = 64;
  std::array<int, dataSize / localSize> in;
  std::array<int, dataSize> out;
  out.fill(0);

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  {
    buffer<int> buff_in{in};
    buffer<int> buff_out{out};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

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

    // Needed implicit group barrier

    q.submit([&](handler &cgh) {
      accessor in(buff_in, cgh, read_only);
      accessor out(buff_out, cgh, write_only, no_init);
      cgh.parallel_for<Kernel>(
          nd_range<1>{{dataSize}, {localSize}}, [=](nd_item<1> i) {
            auto group = i.get_group();
            out[i.get_global_id()] = group_broadcast(
                group, i.get_local_id() == 0 ? in[group.get_group_id(0)] : -1);
          });
    });

    complete_fusion_with_check(fw);
  }

  // Check the results
  for (int i = 0, end = dataSize / localSize; i < end; ++i) {
    int group_id = i / static_cast<int>(localSize);
    assert(out[i] == group_id && "Computation error");
  }

  return 0;
}
