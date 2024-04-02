// RUN: %{build} -fsycl-embed-ir -o %t.out
// RUN: %{run} %t.out

// Test complete_fusion preserves barriers by launching a kernel that requires a
// barrier for correctness.

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/properties/all_properties.hpp>

#include "../helpers.hpp"

using namespace sycl;

class Kernel;

int main() {
  constexpr size_t dataSize = 512;
  constexpr size_t localSize = 64;
  std::array<int, dataSize> in;
  std::array<int, dataSize> out;
  out.fill(0);

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  {
    buffer<int> buff_in{in};
    buffer<int> buff_out{out};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    iota(q, buff_in, 0);

    // Needed implicit group barrier

    q.submit([&](handler &cgh) {
      accessor in(buff_in, cgh, read_only);
      accessor out(buff_out, cgh, write_only, no_init);
      local_accessor<int> lacc(localSize, cgh);
      cgh.parallel_for<Kernel>(
          nd_range<1>{{dataSize}, {localSize}}, [=](nd_item<1> i) {
            auto group = i.get_group();
            if (i.get_local_id() == 0) {
              auto begin = in.begin() + static_cast<int64_t>(
                                            localSize * group.get_group_id(0));
              auto end = begin + localSize;
              std::copy(begin, end, lacc.begin());
            }
            // Test following barrier is preserved
            group_barrier(i.get_group());
            out[i.get_global_id()] = lacc[i.get_local_id()];
          });
    });

    complete_fusion_with_check(fw);
  }

  // Check the results
  for (int i = 0, end = dataSize; i < end; ++i) {
    assert(out[i] == i && "Computation error");
  }

  return 0;
}
