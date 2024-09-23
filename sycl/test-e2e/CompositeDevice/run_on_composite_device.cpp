// RUN: %{build} -o %t.out
// RUN: env ZE_FLAT_DEVICE_HIERARCHY=COMBINED %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/composite_device.hpp>

#ifdef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE

using namespace sycl::ext::oneapi::experimental;

int main() {
  std::vector<sycl::device> CompositeDevs = get_composite_devices();
  for (const auto &Composite : CompositeDevs) {
    // Check that `Composite` is indeed a composite device.
    assert(Composite.has(sycl::aspect::ext_oneapi_is_composite));

    // Create a new context and queue with `Composite` and run a test kernel.
    sycl::context CompositeContext(Composite);
    sycl::queue q(CompositeContext, Composite);
    constexpr size_t N = 1024;
    std::vector<int> TestData(N, 0);
    {
      sycl::buffer TestData_b(TestData.data(), sycl::range<1>{TestData.size()});
      q.submit([&](sycl::handler &cgh) {
        sycl::accessor TestData_acc{TestData_b, cgh};
        cgh.single_task<class TestKernel>([=]() {
          for (size_t i = 0; i < N; ++i)
            TestData_acc[i] = i;
        });
      });
    }
    for (size_t i = 0; i < N; ++i)
      assert(TestData[i] == i);
  }
}

#endif // SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
