// RUN: %{build} -o %t.out
// RUN: env ZE_FLAT_DEVICE_HIERARCHY=COMBINED %{run} %t.out
// REQUIRES: level_zero

#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE

using namespace sycl::ext::oneapi::experimental;

bool isL0Backend(sycl::backend backend) {
  return (backend == sycl::backend::ext_oneapi_level_zero);
}

bool isCombinedMode() {
  const char *Mode = std::getenv("ZE_FLAT_DEVICE_HIERARCHY");
  return (Mode != nullptr) && (std::strcmp(Mode, "COMBINED") == 0);
}

int main() {
  bool IsCombined = isCombinedMode();
  auto Platforms = sycl::platform::get_platforms();

  {
    std::vector<sycl::device> CompositeDevs = get_composite_devices();
    for (const auto &Composite : CompositeDevs) {
      auto Backend = Composite.get_backend();
      auto IsL0 = isL0Backend(Backend);
      // This test requires L0, and it runs with COMBINED mode, check these
      // assumptions.
      assert(IsL0 && IsCombined);

      // Check that `Composite` is indeed a composite device.
      assert(Composite.has(sycl::aspect::ext_oneapi_is_composite));

      // Create a new context and queue with `Composite`.
      sycl::context CompositeContext(Composite);
      sycl::queue q(CompositeContext, Composite);
      constexpr size_t N = 1024;
      std::vector<int> TestData(N, 0);
      {
        sycl::buffer TestData_b(TestData.data(),
                                sycl::range<1>{TestData.size()});
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
}

#endif // SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
