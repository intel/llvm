// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s
//
// XFAIL: hip_nvidia

#include <sycl/detail/core.hpp>
#include <vector>

// Test image-specific printers of the Plugin Interace
//
//CHECK: ---> urEnqueueMemBufferCopyRect(
//CHECK-SAME: .srcOrigin = (struct ur_rect_offset_t){.x = 64, .y = 5, .z = 0}
//CHECK-SAME: .dstOrigin = (struct ur_rect_offset_t){.x = 0, .y = 0, .z = 0}
//CHECK-SAME: .region = (struct ur_rect_region_t){.width = 64, .height = 5, .depth = 1}

using namespace sycl;

int main() {
  constexpr unsigned Width = 16;
  constexpr unsigned Height = 5;
  constexpr unsigned Total = Width * Height;

  std::vector<int> SrcData(Total * 4, 7);
  std::vector<int> DstData(Total, 0);

  queue Queue;
  {
    buffer<int, 2> SrcBuffer(SrcData.data(), range<2>(Height * 2, Width * 2));
    buffer<int, 2> DstBuffer(DstData.data(), range<2>(Height, Width));

    Queue.submit([&](handler &CGH) {
      auto Read = SrcBuffer.get_access<access::mode::read>(
          CGH, range<2>(Height, Width), id(Height, Width));
      auto Write = DstBuffer.get_access<access::mode::write>(CGH);
      CGH.copy(Read, Write);
    });
  }

  // CHECK: ---> urMemBufferPartition(
  // CHECK-SAME: .origin = 128, .size = 32

  constexpr unsigned Size = 64;
  std::vector<int> Data(Size);
  {
    sycl::buffer<int, 1> Buf(Data.data(), Size);
    sycl::buffer<int, 1> SubBuf(Buf, Size / 2, 8);

    Queue.submit([&](sycl::handler &CGH) {
      auto Acc = SubBuf.get_access<sycl::access::mode::write>(CGH);
      CGH.single_task<class empty_task>([=]() {});
    });
  }
  return 0;
}
