// REQUIRES: hip
// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <algorithm>
#include <vector>

class Kernel1Name;
class Kernel2Name;
class Kernel3Name;

template <class TryBodyT>
void checkException(TryBodyT TryBody, const std::string &ExpectedErrMsg) {
  bool ExceptionThrown = false;
  try {
    TryBody();
  } catch (std::exception &E) {
    std::cerr << "Caught: " << E.what() << std::endl;
    std::cerr << "Expect: " << ExpectedErrMsg << std::endl;
    const bool CorrectException =
        std::string(E.what()).find(ExpectedErrMsg) != std::string::npos;
    assert(CorrectException && "Test failed: caught exception is incorrect.");
    ExceptionThrown = true;
  }
  assert(ExceptionThrown && "Expected exception is not thrown");
}

int main() {
  const sycl::device Dev{sycl::default_selector_v};
  const sycl::device Dev2{sycl::default_selector_v};

  const sycl::context Ctx{Dev};
  const sycl::context Ctx2{Dev2};

  sycl::queue Q{Ctx, Dev};
  sycl::queue Q2{Ctx2, Dev2};

  // The code is needed to just have device images in the executable
  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel1Name>([]() {}); });
    Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel2Name>([]() {}); });
  }

  sycl::kernel_id Kernel1ID = sycl::get_kernel_id<Kernel1Name>();
  sycl::kernel_id Kernel2ID = sycl::get_kernel_id<Kernel2Name>();

  {
    sycl::kernel_bundle KernelBundle1 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

    sycl::kernel_bundle KernelBundleCopy = KernelBundle1;
    assert(KernelBundleCopy == KernelBundle1);
    assert(!(KernelBundleCopy != KernelBundle1));
    assert(false == KernelBundle1.empty());
    assert(Ctx.get_platform().get_backend() == KernelBundle1.get_backend());
    assert(KernelBundle1.get_context() == Ctx);
    assert(KernelBundle1.get_devices() == (std::vector<sycl::device>){Dev});
    assert(KernelBundle1.has_kernel(Kernel1ID));
    assert(KernelBundle1.has_kernel(Kernel2ID));
    assert(KernelBundle1.has_kernel(Kernel1ID, Dev));
    assert(KernelBundle1.has_kernel(Kernel2ID, Dev));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel1ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel1ID);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel2ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel1ID,
         &Dev](const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel1ID, Dev);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel2ID,
         &Dev](const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID, Dev);
        }));

    assert(sycl::has_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev}));
    assert(sycl::has_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                              {Kernel2ID}));

    // no duplicate devices
    sycl::kernel_bundle KernelBundleDupTest =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev, Dev},
                                                           {Kernel1ID});
    assert(KernelBundleDupTest.get_devices().size() == 1);
  }

  // The following check relies on "-fsycl-device-code-split=per_kernel" option,
  // so it is expected that each kernel in a separate device image.
  // Verify that get_kernel_bundle filters out device images based on vector
  // of kernel_id's and Selector.

  {
    // Test get_kernel_bundle with filters, join and get_kernel_ids API.
    sycl::kernel_bundle KernelBundleInput1 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                           {Kernel1ID});
    assert(KernelBundleInput1.has_kernel(Kernel1ID));
    assert(!KernelBundleInput1.has_kernel(Kernel2ID));

    auto Selector =
        [&Kernel2ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID);
        };

    sycl::kernel_bundle KernelBundleInput2 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                           Selector);
    assert(!KernelBundleInput2.has_kernel(Kernel1ID));
    assert(KernelBundleInput2.has_kernel(Kernel2ID));

    sycl::kernel_bundle KernelBundleJoint =
        sycl::join(std::vector<sycl::kernel_bundle<sycl::bundle_state::input>>{
            KernelBundleInput1, KernelBundleInput2});

    assert(KernelBundleJoint.has_kernel(Kernel1ID));
    assert(KernelBundleJoint.has_kernel(Kernel2ID));

    std::vector<sycl::kernel_id> KernelIDs = KernelBundleJoint.get_kernel_ids();

    assert(KernelIDs.size() == 2);
  }

  {
    // Error handling

    // The error handling tests rely on Kernel3Name binary existing.
    sycl::buffer<int, 1> Buf(sycl::range<1>{1});
    Q.submit([&](sycl::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::write>(CGH);
      CGH.single_task<Kernel3Name>([=]() { Acc[0] = 42; });
    });

    {
      sycl::host_accessor HostAcc(Buf, sycl::write_only);
      assert(HostAcc[0] == 42);
    }

    std::cerr << "Empty list of devices for get_kernel_bundle" << std::endl;
    checkException(
        [&]() {
          sycl::get_kernel_bundle<sycl::bundle_state::input>(
              Ctx, std::vector<sycl::device>{});
        },
        "Not all devices are associated with the context or vector of devices "
        "is empty");

    std::cerr << "Mismatched contexts for join" << std::endl;
    checkException(
        [&]() {
          sycl::kernel_bundle KernelBundleObject1 =
              sycl::get_kernel_bundle<sycl::bundle_state::object>(Ctx);

          sycl::kernel_bundle KernelBundleObject2 =
              sycl::get_kernel_bundle<sycl::bundle_state::object>(Ctx2);

          sycl::join(
              std::vector<sycl::kernel_bundle<sycl::bundle_state::object>>{
                  KernelBundleObject1, KernelBundleObject2});
        },
        "Not all input bundles have the same associated context");

    std::cerr << "Not found kernel" << std::endl;
    checkException(
        [&]() {
          sycl::kernel_id Kernel3ID = sycl::get_kernel_id<Kernel3Name>();
          sycl::kernel_bundle KernelBundleExecutable =
              sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                  Ctx, {Dev}, {Kernel3ID});

          KernelBundleExecutable.get_kernel(Kernel1ID);
        },
        "The kernel bundle does not contain the kernel identified by kernelId");

    std::cerr << "Empty devices for has_kernel_bundle" << std::endl;
    checkException(
        [&]() {
          sycl::has_kernel_bundle<sycl::bundle_state::input>(
              Ctx, std::vector<sycl::device>{});
        },
        "Not all devices are associated with the context or vector of devices "
        "is empty");
  }

  return 0;
}
