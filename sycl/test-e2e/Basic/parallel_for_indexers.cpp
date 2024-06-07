// RUN: %{build} -o %t2.out
// RUN: %{run} %t2.out

#include <sycl/detail/core.hpp>
#include <sycl/types.hpp>

#include <cassert>
#include <memory>

using namespace sycl;

// TODO add cases with dimensions more than 1
int main() {
  // Id indexer
  {
    std::vector<int> data(10, -1);
    const range<1> globalRange(6);
    {
      buffer<int, 1> b(data.data(), range<1>(10),
                       {property::buffer::use_host_ptr()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class id1>(globalRange,
                                    [=](id<1> index) { B[index] = index[0]; });
      });
    }
    for (int i = 0; i < data.size(); i++) {
      const int id = data[i];
      if (i < globalRange[0]) {
        assert(id == i);
      } else {
        assert(id == -1);
      }
    }
  }
  // Item indexer without offset
  {
    std::vector<int2> data(10, int2{-1});
    const range<1> globalRange(6);
    {
      buffer<int2, 1> b(data.data(), range<1>(10),
                        {property::buffer::use_host_ptr()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class item1_nooffset>(
            globalRange, [=](item<1, false> index) {
              B[index.get_id()] = int2{index.get_id()[0], index.get_range()[0]};
            });
      });
    }
    for (int i = 0; i < data.size(); i++) {
      const int id = data[i].s0();
      const int range = data[i].s1();
      if (i < globalRange[0]) {
        assert(id == i);
        assert(range == globalRange[0]);
      } else {
        assert(id == -1);
        assert(range == -1);
      }
    }
  }
  // Item indexer with offset
  {
    std::vector<int3> data(10, int3{-1});
    const range<1> globalRange(6);
    const id<1> globalOffset(4);
    {
      buffer<int3, 1> b(data.data(), range<1>(10),
                        {property::buffer::use_host_ptr()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class item1_offset>(
            globalRange, globalOffset, [=](item<1> index) {
              B[index.get_id()] = int3{index.get_id()[0], index.get_range()[0],
                                       index.get_offset()[0]};
            });
      });
    }
    for (int i = 0; i < data.size(); i++) {
      const int id = data[i].s0();
      const int range = data[i].s1();
      const int offset = data[i].s2();
      if (i < globalOffset[0]) {
        assert(id == -1);
        assert(range == -1);
        assert(offset == -1);
      } else {
        assert(id == i);
        assert(range == globalRange[0]);
        assert(offset == globalOffset[0]);
      }
    }
  }
  // ND_Item indexer
  {
    std::vector<int3> data(10, int3{-1});
    const range<1> globalRange(6);
    const range<1> localRange(3);
    const id<1> globalOffset(4);
    const nd_range<1> ndRange(globalRange, localRange, globalOffset);
    {
      buffer<int3, 1> b(data.data(), range<1>(10),
                        {property::buffer::use_host_ptr()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class item1_nd_range>(ndRange, [=](nd_item<1> index) {
          B[index.get_global_id()] =
              int3{index.get_global_id()[0], index.get_global_range()[0],
                   index.get_offset()[0]};
        });
      });
    }
    for (int i = 0; i < data.size(); i++) {
      const int id = data[i].s0();
      const int range = data[i].s1();
      const int offset = data[i].s2();
      if (i < globalOffset[0]) {
        assert(id == -1);
        assert(range == -1);
        assert(offset == -1);
      } else {
        assert(id == i);
        assert(range == globalRange[0]);
        assert(offset == globalOffset[0]);
      }
    }
  }
  return 0;
}
