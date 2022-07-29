// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER
//
// XFAIL: hip_nvidia

#include <sycl/sycl.hpp>
#include <vector>

// Test image-specific printers of the Plugin Interace
//
//CHECK: ---> piEnqueueMemBufferCopyRect(
//CHECK:   pi_buff_rect_offset x_bytes/y/z : 64/5/0
//CHECK:   pi_buff_rect_offset x_bytes/y/z : 0/0/0
//CHECK:   pi_buff_rect_region width_bytes/height/depth : 64/5/1
//CHECK:   pi_buff_rect_region width_bytes/height/depth : 64/5/1

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

  // CHECK: ---> piMemBufferPartition(
  // CHECK:   pi_buffer_region origin/size : 128/32

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
