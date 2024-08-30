// RUN: %{build} %{embed-ir} -o %t.out
// RUN: %{run} %t.out

// Test fusion works with permute and remapping.

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

#include "../helpers.hpp"
#include "sycl/group_algorithm.hpp"

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
  // Needed to check results
  size_t sg_size = 0;

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};
  {
    buffer<int> buff_in{in};
    buffer<int> buff_out0{out0};
    buffer<int> buff_out1{out1};
    buffer<size_t> buff_sg_size{&sg_size, 1};

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
      accessor sg_size(buff_sg_size, cgh, write_only, no_init);
      accessor in(buff_in, cgh, read_only);
      accessor out(buff_out0, cgh, write_only, no_init);
      cgh.parallel_for<Kernel0>(
          nd_range<1>{{dataSize}, {localSize}}, [=](nd_item<1> i) {
            sub_group group = i.get_sub_group();
            int gid = i.get_global_id();
            int sgid = group.get_group_id();
            sg_size[0] = group.get_max_local_range()[0];
            out[gid] = permute_group_by_xor(
                group, gid, sgid % group.get_max_local_range()[0]);
          });
    });

    q.submit([&](handler &cgh) {
      accessor in(buff_in, cgh, read_only);
      accessor out(buff_out1, cgh, write_only, no_init);
      cgh.parallel_for<Kernel1>(
          nd_range<1>{{dataSize / 2}, {localSize}}, [=](nd_item<1> i) {
            sub_group group = i.get_sub_group();
            int gid = i.get_global_id();
            int sgid = group.get_group_id();
            out[gid] = permute_group_by_xor(
                group, gid, sgid % group.get_max_local_range()[0]);
          });
    });

    complete_fusion_with_check(fw);
  }

  // Check the results
  int SGid = 0;
  int SGLid = 0;
  int SGBeginGid = 0;
  int j = 0;
  const auto check = [sg_size, &SGid, &SGLid, &SGBeginGid, &out0,
                      &out1](int j, bool checkSmall) {
    if (j % localSize % sg_size == 0) {
      SGid++;
      SGLid = 0;
      SGBeginGid = j;
    }
    if (j % localSize == 0) {
      SGid = 0;
      SGLid = 0;
      SGBeginGid = j;
    }
    assert(out0[j] == SGBeginGid + (SGLid ^ (SGid % sg_size)));
    assert(!checkSmall || (out1[j] == SGBeginGid + (SGLid ^ (SGid % sg_size))));
    SGLid++;
  };
  for (int end = dataSize / 2; j < end; j++) {
    check(j, true);
  }
  for (int end = dataSize; j < end; j++) {
    check(j, false);
  }

  return 0;
}
