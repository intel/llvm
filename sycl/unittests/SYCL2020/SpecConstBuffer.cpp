#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <CL/sycl.hpp>
#include <detail/device_image_impl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>


class TestKernelNative;

const static sycl::specialization_id<int> SpecConst2{42};
__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<TestKernelNative> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() {
    return "SpecializationConstant_TestKernelNative";
  }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

template <> const char *get_spec_constant_symbolic_ID<SpecConst2>() {
  return "SC2";
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)


static sycl::unittest::PiImage generateImageWithSpecConstsNative() {
  using namespace sycl::unittest;

  std::vector<char> SpecConstData;
  PiProperty SC1 = makeSpecConstant<int>(SpecConstData, "SC1", {0}, {0}, {42});
  PiProperty SC2 = makeSpecConstant<int>(SpecConstData, "SC2", {1}, {0}, {8});

  PiPropertySet PropSet;
  addSpecConstants({SC1, SC2}, std::move(SpecConstData), PropSet);

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries =
      makeEmptyKernels({"SpecializationConstant_TestKernelNative"});

  PiImage Img{PI_DEVICE_BINARY_TYPE_NATIVE,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

std::atomic<int> RedefinedMemReleaseCalled;
static pi_result redefinedMemRelease(pi_mem mem) {
	RedefinedMemReleaseCalled = 1;
	std::cout << "rdfMemRel" << std::endl;
	return PI_SUCCESS;
}

pi_result redefinedProgramSetSpecializationConstant(pi_program prog, pi_uint32 spec_id,
                                      size_t spec_size, const void *spec_value) {
  return PI_SUCCESS;
}
static sycl::unittest::PiImage Img { generateImageWithSpecConstsNative() };
static sycl::unittest::PiImageArray<1> ImgArray{&Img};




TEST(SpecConstBuffer, ResourceCleanUp) {
  RedefinedMemReleaseCalled = 0;
  {
    sycl::platform Plt{sycl::default_selector()};
    if (Plt.is_host()) {
      std::cerr << "Test is not supported on host, skipping\n";
      return; // test is not supported on host.
    }

    if (Plt.get_backend() == sycl::backend::ext_oneapi_cuda) {
      std::cerr << "Test is not supported on CUDA platform, skipping\n";
      return;
    }

    if (Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
      std::cerr << "Test is not supported on HIP platform, skipping\n";
      return;
    }

    sycl::unittest::PiMock Mock{Plt};
    setupDefaultMockAPIs(Mock);
    Mock.redefine<sycl::detail::PiApiKind::piMemRelease>(redefinedMemRelease);
    Mock.redefine<sycl::detail::PiApiKind::piextProgramSetSpecializationConstant>(redefinedProgramSetSpecializationConstant);
    const sycl::device Dev = Plt.get_devices()[0];
    sycl::queue Q{Dev};
    const sycl::context Ctx = Q.get_context();
    std::vector<sycl::kernel_id> kernelId = {sycl::get_kernel_id<TestKernelNative>()};
    sycl::kernel_bundle KernelBundle =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, kernelId);
    KernelBundle.set_specialization_constant<SpecConst2>(1);
    auto exeBundle = sycl::build(KernelBundle);
    Q.submit([&](sycl::handler &CGH) {
      CGH.use_kernel_bundle(exeBundle);
      // CGH.set_specialization_constant<SpecConst2>(1);
      // CGH.single_task<TestKernel>([=](sycl::kernel_handler KH) {
      //   (void)KH.get_specialization_constant<SpecConst1>();
      // });
      CGH.single_task<TestKernelNative>([]() {});
    });
    Q.wait();
  }
  EXPECT_EQ(RedefinedMemReleaseCalled, 1);
}

