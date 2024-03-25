// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
using namespace sycl;

int main() {
  queue testQueue;

  char testResult;
  {
    buffer<char, 1> resultBuf(&testResult, range<1>(1));

    int32_t data = 42;
    buffer<int32_t, 1> buf(&data, range<1>(1));

    testQueue.submit([&](handler &cgh) {
      auto globAcc = buf.template get_access<access::mode::atomic>(cgh);
      auto resultAcc = resultBuf.template get_access<access::mode::write>(cgh);
      cgh.single_task<class foo>([=]() {
        auto a = globAcc[0];
        char result = 0;
        int32_t expected = 0; // Set to wrong value.
        char updated = a.compare_exchange_strong(expected, 1);
        if (updated)
          // Update shouldn't happen, value in memory is different from
          // expected!
          result = 1;
        if (expected != 42)
          // "Expected" inout parameter wasn't updated to the value read!
          result = 1;
        resultAcc[0] = result;
      });
    });
  }
  assert(testResult == 0 && "Test failed!");
  return testResult;
}
