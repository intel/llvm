// UNSUPPORTED: hip_nvidia
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER

#include <iostream>
#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr long width = 16;
constexpr long height = 5;
constexpr long total = width * height;

constexpr long depth = 3;
constexpr long total3D = total * depth;

void remind() {
  /*
    https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBufferRect.html

    buffer_origin defines the (x, y, z) offset in the memory region associated
    with buffer. For a 2D rectangle region, the z value given by
    buffer_origin[2] should be 0. The offset in bytes is computed as
    buffer_origin[2] × buffer_slice_pitch + buffer_origin[1] × buffer_row_pitch
    + buffer_origin[0].

    region defines the (width in bytes, height in rows, depth in slices) of the
    2D or 3D rectangle being read or written. For a 2D rectangle copy, the depth
    value given by region[2] should be 1. The values in region cannot be 0.


    buffer_row_pitch is the length of each row in bytes to be used for the
    memory region associated with buffer. If buffer_row_pitch is 0,
    buffer_row_pitch is computed as region[0].

    buffer_slice_pitch is the length of each 2D slice in bytes to be used for
    the memory region associated with buffer. If buffer_slice_pitch is 0,
    buffer_slice_pitch is computed as region[1] × buffer_row_pitch.
  */
  std::cout << "For BUFFERS" << std::endl;
  std::cout << "         Region SHOULD be : " << width * sizeof(float) << "/"
            << height << "/" << depth << std::endl; // 64/5/3
  std::cout << "  RowPitch SHOULD be 0 or : " << width * sizeof(float)
            << std::endl; // 0 or 64
  std::cout << "SlicePitch SHOULD be 0 or : " << width * sizeof(float) * height
            << std::endl
            << std::endl; // 0 or 320
}
// ----------- FUNCTIONAL
template <template <int> class T> static void printRangeId(T<3> arr) {
  std::cout << ":: "
            << "{" << arr[0] << ", " << arr[1] << ", " << arr[2] << "}"
            << std::endl;
}

void testDetailConvertToArrayOfN() {
  // ranges, as used with buffers (args reverse order for images)
  range<1> range_1D(width);
  range<2> range_2D(height, width);
  range<3> range_3D(depth, height, width);

  range<3> arr1 = sycl::detail::convertToArrayOfN<3, 1>(range_1D);
  // {16,1,1}
  printRangeId(arr1);
  assert(arr1[0] == width && arr1[1] == 1 && arr1[2] == 1 &&
         "arr1 expected as {16,1,1}");

  range<3> arr2 = sycl::detail::convertToArrayOfN<3, 1>(range_2D);
  //{5, 16, 1}
  printRangeId(arr2);
  assert(arr2[0] == height && arr2[1] == width && arr2[2] == 1 &&
         "arr2 expected as {5, 16, 1}");

  range<3> arr3 = sycl::detail::convertToArrayOfN<3, 1>(range_3D);
  //{3, 5, 16}
  printRangeId(arr3);
  assert(arr3[0] == depth && arr3[1] == height && arr3[2] == width &&
         "arr3 expected as {3,5,16}");

  range<2> smaller2 = sycl::detail::convertToArrayOfN<2, 1>(range_3D);
  //{3,5}
  std::cout << "{" << smaller2[0] << "," << smaller2[1] << "}" << std::endl;
  assert(smaller2[0] == depth && smaller2[1] == height &&
         "smaller2 expected {3,5} ");

  range<1> smaller1 = sycl::detail::convertToArrayOfN<1, 1>(range_3D);
  //{3}
  assert(smaller1[0] == depth && "smaller1 expected {3} ");
}

// class to give access to protected function getLinearIndex
template <typename T, int Dims>
class AccTest : public accessor<T, Dims, access::mode::read_write,
                                access::target::host_buffer,
                                access::placeholder::false_t> {
  using AccessorT =
      accessor<T, Dims, access::mode::read_write, access::target::host_buffer,
               access::placeholder::false_t>;

public:
  AccTest(AccessorT acc) : AccessorT(acc) {}

  size_t gLI(id<Dims> idx) { return AccessorT::getLinearIndex(idx); }
};

void testGetLinearIndex() {
  constexpr int x = 4, y = 3, z = 1;
  // width=16, height=5, depth = 3.
  // row is 16 (ie. width)
  // slice is 80 (ie width * height)
  size_t target_1D = x;
  size_t target_2D = (y * width) + x; // s.b. (3*16) + 4 => 52
  size_t target_3D =
      (height * width * z) + (y * width) + x; // s.b. 80 + (3*16) + 4 => 132

  std::vector<float> data_1D(width, 13);
  std::vector<float> data_2D(total, 7);
  std::vector<float> data_3D(total3D, 17);

  // test accessor protected function
  {
    buffer<float, 1> buffer_1D(data_1D.data(), range<1>(width));
    buffer<float, 2> buffer_2D(data_2D.data(), range<2>(height, width));
    buffer<float, 3> buffer_3D(data_3D.data(), range<3>(depth, height, width));

    auto acc_1D = buffer_1D.get_access<access::mode::read_write>();
    auto accTest_1D = AccTest<float, 1>(acc_1D);
    size_t linear_1D = accTest_1D.gLI(id<1>(x)); // s.b. 4
    std::cout << "linear_1D: " << linear_1D << "  target_1D: " << target_1D
              << std::endl;
    assert(linear_1D == target_1D && "linear_1D s.b. 4");

    auto acc_2D = buffer_2D.get_access<access::mode::read_write>();
    auto accTest_2D = AccTest<float, 2>(acc_2D);
    size_t linear_2D = accTest_2D.gLI(id<2>(y, x));
    std::cout << "linear_2D: " << linear_2D << "  target_2D: " << target_2D
              << std::endl;
    assert(linear_2D == target_2D && "linear_2D s.b. 52");

    auto acc_3D = buffer_3D.get_access<access::mode::read_write>();
    auto accTest_3D = AccTest<float, 3>(acc_3D);
    size_t linear_3D = accTest_3D.gLI(id<3>(z, y, x));
    std::cout << "linear_3D: " << linear_3D << "  target_3D: " << target_3D
              << std::endl;
    assert(linear_3D == target_3D && "linear_3D s.b. 132");
  }

  // common.hpp variant of getLinearIndex
  size_t lin_1D = getLinearIndex(id<1>(x), range<1>(width));
  std::cout << "lin_1D: " << lin_1D << std::endl;
  assert(lin_1D == target_1D && "lin_1D s.b. 4");

  size_t lin_2D = getLinearIndex(id<2>(y, x), range<2>(height, width));
  std::cout << "lin_2D: " << lin_2D << "  target_2D: " << target_2D
            << std::endl;
  assert(lin_2D == target_2D && "lin_2D s.b. 52");

  size_t lin_3D =
      getLinearIndex(id<3>(z, y, x), range<3>(depth, height, width));
  std::cout << "lin_3D: " << lin_3D << "  target_3D: " << target_3D
            << std::endl;
  assert(lin_3D == target_3D && "lin_3D s.b. 132");
}

// ----------- BUFFERS

void testcopyD2HBuffer() {
  std::cout << "start copyD2H-buffer" << std::endl;
  std::vector<float> data_from_1D(width, 13);
  std::vector<float> data_to_1D(width, 0);
  std::vector<float> data_from_2D(total, 7);
  std::vector<float> data_to_2D(total, 0);
  std::vector<float> data_from_3D(total3D, 17);
  std::vector<float> data_to_3D(total3D, 0);

  {
    buffer<float, 1> buffer_from_1D(data_from_1D.data(), range<1>(width));
    buffer<float, 1> buffer_to_1D(data_to_1D.data(), range<1>(width));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyD2H_1D>(
          buffer_from_1D.get_range(),
          [=](id<1> index) { write[index] = read[index] * -1; });
    });
  } // ~buffer 1D

  {
    buffer<float, 2> buffer_from_2D(data_from_2D.data(),
                                    range<2>(height, width));
    buffer<float, 2> buffer_to_2D(data_to_2D.data(), range<2>(height, width));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyD2H_2D>(
          buffer_from_2D.get_range(),
          [=](id<2> index) { write[index] = read[index] * -1; });
    });
  } // ~buffer 2D

  {
    buffer<float, 3> buffer_from_3D(data_from_3D.data(),
                                    range<3>(depth, height, width));
    buffer<float, 3> buffer_to_3D(data_to_3D.data(),
                                  range<3>(depth, height, width));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyD2H_3D>(
          buffer_from_3D.get_range(),
          [=](id<3> index) { write[index] = read[index] * -1; });
    });
  } // ~buffer 3D

  std::cout << "end copyD2H-buffer" << std::endl;
}

void testcopyH2DBuffer() {
  // copy between two queues triggers a piEnqueueMemBufferMap followed by
  // copyH2D, followed by a copyD2H, followed by a piEnqueueMemUnmap
  // Here we only care about checking copyH2D

  std::cout << "start copyH2D-buffer" << std::endl;
  std::vector<float> data_from_1D(width, 13);
  std::vector<float> data_to_1D(width, 0);
  std::vector<float> data_from_2D(total, 7);
  std::vector<float> data_to_2D(total, 0);
  std::vector<float> data_from_3D(total3D, 17);
  std::vector<float> data_to_3D(total3D, 0);

  {
    buffer<float, 1> buffer_from_1D(data_from_1D.data(), range<1>(width));
    buffer<float, 1> buffer_to_1D(data_to_1D.data(), range<1>(width));

    device Dev{default_selector{}};
    context myCtx{Dev};
    context otherCtx{Dev};

    queue myQueue{myCtx, Dev};
    queue otherQueue{otherCtx, Dev};
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_1D>(
          buffer_from_1D.get_range(),
          [=](id<1> index) { write[index] = read[index] * -1; });
    });
    myQueue.wait();

    otherQueue.submit([&](handler &cgh) {
      auto read = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_1D_2nd>(
          buffer_from_1D.get_range(),
          [=](id<1> index) { write[index] = read[index] * 10; });
    });
  } // ~buffer 1D

  {
    buffer<float, 2> buffer_from_2D(data_from_2D.data(),
                                    range<2>(height, width));
    buffer<float, 2> buffer_to_2D(data_to_2D.data(), range<2>(height, width));

    device Dev{default_selector{}};
    context myCtx{Dev};
    context otherCtx{Dev};

    queue myQueue{myCtx, Dev};
    queue otherQueue{otherCtx, Dev};
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_2D>(
          buffer_from_2D.get_range(),
          [=](id<2> index) { write[index] = read[index] * -1; });
    });

    otherQueue.submit([&](handler &cgh) {
      auto read = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_2D_2nd>(
          buffer_from_2D.get_range(),
          [=](id<2> index) { write[index] = read[index] * 10; });
    });
  } // ~buffer 22

  {
    buffer<float, 3> buffer_from_3D(data_from_3D.data(),
                                    range<3>(depth, height, width));
    buffer<float, 3> buffer_to_3D(data_to_3D.data(),
                                  range<3>(depth, height, width));

    device Dev{default_selector{}};
    context myCtx{Dev};
    context otherCtx{Dev};

    queue myQueue{myCtx, Dev};
    queue otherQueue{otherCtx, Dev};
    myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_3D>(
          buffer_from_3D.get_range(),
          [=](id<3> index) { write[index] = read[index] * -1; });
    });

    otherQueue.submit([&](handler &cgh) {
      auto read = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class copyH2D_3D_2nd>(
          buffer_from_3D.get_range(),
          [=](id<3> index) { write[index] = read[index] * 10; });
    });
  } // ~buffer 3D

  std::cout << "end copyH2D-buffer" << std::endl;
}

void testcopyD2DBuffer() {
  std::cout << "start copyD2D-buffer" << std::endl;
  std::vector<float> data_from_1D(width, 13);
  std::vector<float> data_to_1D(width, 0);
  std::vector<float> data_from_2D(total, 7);
  std::vector<float> data_to_2D(total, 0);
  std::vector<float> data_from_3D(total3D, 17);
  std::vector<float> data_to_3D(total3D, 0);
  {
    buffer<float, 1> buffer_from_1D(data_from_1D.data(), range<1>(width));
    buffer<float, 1> buffer_to_1D(data_to_1D.data(), range<1>(width));
    buffer<float, 2> buffer_from_2D(data_from_2D.data(),
                                    range<2>(height, width));
    buffer<float, 2> buffer_to_2D(data_to_2D.data(), range<2>(height, width));
    buffer<float, 3> buffer_from_3D(data_from_3D.data(),
                                    range<3>(depth, height, width));
    buffer<float, 3> buffer_to_3D(data_to_3D.data(),
                                  range<3>(depth, height, width));

    queue myQueue;
    auto e1 = myQueue.submit([&](handler &cgh) {
      auto read = buffer_from_1D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_1D.get_access<access::mode::write>(cgh);
      cgh.copy(read, write);
    });
    auto e2 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e1);
      auto read = buffer_from_2D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_2D.get_access<access::mode::write>(cgh);
      cgh.copy(read, write);
    });
    auto e3 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e2);
      auto read = buffer_from_3D.get_access<access::mode::read>(cgh);
      auto write = buffer_to_3D.get_access<access::mode::write>(cgh);
      cgh.copy(read, write);
    });

  } // ~buffer
  std::cout << "end copyD2D-buffer" << std::endl;
}

void testFill_Buffer() {
  std::cout << "start testFill Buffer" << std::endl;
  std::vector<float> data_1D(width, 0);
  std::vector<float> data_2D(total, 0);
  std::vector<float> data_3D(total3D, 0);
  {
    buffer<float, 1> buffer_1D(data_1D.data(), range<1>(width));
    buffer<float, 2> buffer_2D(data_2D.data(), range<2>(height, width));
    buffer<float, 3> buffer_3D(data_3D.data(), range<3>(depth, height, width));

    queue myQueue;
    auto e1 = myQueue.submit([&](handler &cgh) {
      auto acc1D = buffer_1D.get_access<sycl::access::mode::write>(cgh);
      cgh.fill(acc1D, float{1});
    });
    auto e2 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e1);
      auto acc2D = buffer_2D.get_access<sycl::access::mode::write>(cgh);
      cgh.fill(acc2D, float{2});
    });
    auto e3 = myQueue.submit([&](handler &cgh) {
      cgh.depends_on(e2);
      auto acc3D = buffer_3D.get_access<sycl::access::mode::write>(cgh);
      cgh.fill(acc3D, float{3});
    });
  } // ~buffer
  std::cout << "end testFill Buffer" << std::endl;
}

// --------------

int main() {
  remind();

  testDetailConvertToArrayOfN();
  testGetLinearIndex();

  testcopyD2HBuffer();
  testcopyH2DBuffer();
  testcopyD2DBuffer();
  testFill_Buffer();
}

// ----------- BUFFERS

// CHECK-LABEL: start copyD2H-buffer
// CHECK: ---> piEnqueueMemBufferRead(
// CHECK: <unknown> : 64
// CHECK: ---> piEnqueueMemBufferReadRect(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
// CHECK-NEXT: <unknown> : 64
// CHECK: ---> piEnqueueMemBufferReadRect(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK: end copyD2H-buffer

// CHECK-LABEL: start copyH2D-buffer
// CHECK: ---> piEnqueueMemBufferWrite(
// CHECK: <unknown> : 64
// CHECK:  ---> piEnqueueMemBufferWriteRect(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 0
// CHECK-NEXT: <unknown> : 64
// CHECK:  ---> piEnqueueMemBufferWriteRect(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK: end copyH2D-buffer

// CHECK-LABEL: start copyD2D-buffer
// CHECK: ---> piEnqueueMemBufferCopy(
// CHECK: <unknown> : 64
// CHECK: ---> piEnqueueMemBufferCopyRect(
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/3
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK-NEXT: <unknown> : 64
// CHECK-NEXT: <unknown> : 320
// CHECK: end copyD2D-buffer

// CHECK-LABEL: start testFill Buffer
// CHECK: ---> piEnqueueMemBufferFill(
// CHECK: <unknown> : 4
// CHECK-NEXT: <unknown> : 0
// CHECK-NEXT: <unknown> : 64
// CHECK: end testFill Buffer
