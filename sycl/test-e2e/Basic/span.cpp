// REQUIRES: usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Fails to release USM pointer on HIP for NVIDIA
// XFAIL: hip_nvidia

#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

void testSpanCapture() {
  // This test creates spans that are backed by USM.
  // ensures they can be captured by device lambda
  // and that read and write operations function correctly
  // across capture.
  queue Q;

  constexpr long numReadTests = 2;
  const range<1> NumberOfReadTestsRange(numReadTests);
  buffer<int, 1> SpanRead(NumberOfReadTestsRange);

  // span from a vector
  // We will create a vector, backed by a USM allocator. And a span from that.
  using vec_alloc = usm_allocator<int, usm::alloc::shared>;
  // Create allocator for device associated with q
  vec_alloc myAlloc(Q);
  // Create std vector with the allocator
  std::vector<int, vec_alloc> vecUSM(4, myAlloc);
  std::iota(vecUSM.begin(), vecUSM.end(), 1);
  sycl::span<int> vecUSM_span{vecUSM};
  vecUSM_span[0] += 100; // 101  modify first value using span affordance.

  // span from USM memory
  auto *usm_data = malloc_shared<int>(4, Q);
  sycl::span<int> usm_span(usm_data, 4);
  std::iota(usm_span.begin(), usm_span.end(), 1);
  usm_span[0] += 100; // 101 modify first value using span affordance.

  event E = Q.submit([&](handler &cgh) {
    auto can_read_from_span_acc = SpanRead.get_access<access::mode::write>(cgh);
    cgh.single_task<class hi>([=] {
      // read from the spans.
      can_read_from_span_acc[0] = vecUSM_span[0];
      can_read_from_span_acc[1] = usm_span[0];

      // write to the spans
      vecUSM_span[1] += 1000;
      usm_span[1] += 1000;
    });
  });
  E.wait();

  // check out the read operations, should have gotten 101 from each
  host_accessor can_read_from_span_acc(SpanRead, read_only);
  for (int i = 0; i < numReadTests; i++) {
    assert(can_read_from_span_acc[i] == 101 &&
           "read check should have gotten 100");
  }

  // were the spans successfully modified via write?
  assert(vecUSM_span[1] == 1002 &&
         "vecUSM_span write check should have gotten 1001");
  assert(usm_span[1] == 1002 && "usm_span write check should have gotten 1001");

  free(usm_data, Q);
}

void set_all_span_values(sycl::span<int> container, int v) {
  for (auto &e : container)
    e = v;
}

void testSpanOnDevice() {
  // this test creates a simple span on device,
  // passes it to a function that operates on it
  // and ensures it worked correctly
  queue Q;
  constexpr long numReadTests = 4;
  const range<1> NumberOfReadTestsRange(numReadTests);
  buffer<int, 1> SpanRead(NumberOfReadTestsRange);

  event E = Q.submit([&](handler &cgh) {
    auto can_read_from_span_acc = SpanRead.get_access<access::mode::write>(cgh);
    cgh.single_task<class ha>([=] {
      // create a span on device, pass it to function that modifies it
      // read values back out.
      int a[]{1, 2, 3, 4};
      sycl::span<int> a_span{a};
      set_all_span_values(a_span, 10);
      for (int i = 0; i < numReadTests; i++)
        can_read_from_span_acc[i] = a_span[i];
    });
  });
  E.wait();

  // check out the read operations, should have gotten 10 from each
  host_accessor can_read_from_span_acc(SpanRead, read_only);
  for (int i = 0; i < numReadTests; i++) {
    assert(can_read_from_span_acc[i] == 10 &&
           "read check should have gotten 10");
  }
}

int main() {
  testSpanCapture();
  testSpanOnDevice();

  return 0;
}
