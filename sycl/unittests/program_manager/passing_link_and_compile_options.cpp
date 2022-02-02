
#include <CL/sycl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

std::string current_link_options, current_compile_options;

class EAMTestKernel;
const char EAMTestKernelName[] = "EAMTestKernel";
constexpr unsigned EAMTestKernelNumArgs = 4;

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <> struct KernelInfo<EAMTestKernel> {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return EAMTestKernelName; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)


static sycl::unittest::PiImage generateEAMTestKernelImage(std::string _cmplOptions = "", std::string _lnkOptions = "") {
  using namespace sycl::unittest;

  // Eliminated arguments are 1st and 3rd.
  std::vector<unsigned char> KernelEAM{0b00000101};
  PiProperty EAMKernelPOI = makeKernelParamOptInfo(
      EAMTestKernelName, EAMTestKernelNumArgs, KernelEAM);
  PiArray<PiProperty> ImgKPOI{std::move(EAMKernelPOI)};

  PiPropertySet PropSet;
  PropSet.insert(__SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO,
                 std::move(ImgKPOI));

  std::vector<unsigned char> Bin{0, 1, 2, 3, 4, 5}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({EAMTestKernelName});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              _cmplOptions.c_str(),                   // Compile options
              _lnkOptions.c_str(),                    // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}


static pi_result redefinedProgramLink(pi_context, pi_uint32, const pi_device *,
                                      const char * _linkOpts, pi_uint32,
                                      const pi_program *,
                                      void (*)(pi_program, void *), void *,
                                      pi_program *) {
  if (_linkOpts) {
      current_link_options = std::string(_linkOpts);
  }
  return PI_SUCCESS;
}

static pi_result redefinedProgramCompile(pi_program, pi_uint32,
                                         const pi_device *, const char * _compileOpts,
                                         pi_uint32, const pi_program *,
                                         const char **,
                                         void (*)(pi_program, void *), void *) {
  if (_compileOpts) {
      current_compile_options = std::string(_compileOpts);
  }
  return PI_SUCCESS;
}

Mock.redefine<PiApiKind::piProgramCompile>(redefinedProgramCompile);
Mock.redefine<PiApiKind::piProgramLink>(redefinedProgramLink);

TEST(Link_Compile_Options, linkOptionsTest_empty) {
    expected_link_options.clear();
    expected_compile_options.clear();
    std::string expected_compile_options = "";
    std::string expected_link_options = "";
    static sycl::unittest::PiImage DevImage = generateEAMTestKernelImage(expected_compile_options, expected_link_options);
    auto BundleObj = sycl::compile(DevImage);
    sycl::link(BundleObj);
    EXPECT_EQ(expected_link_options, current_link_options);
}

TEST(Link_Compile_Options, linkOptionsTest_one_param) {
    expected_link_options.clear();
    expected_compile_options.clear();
    std::string expected_compile_options = "";
    std::string expected_link_options = "-foo";
    static sycl::unittest::PiImage DevImage = generateEAMTestKernelImage(expected_compile_options, expected_link_options);
    auto BundleObj = sycl::compile(DevImage);
    sycl::link(BundleObj);
    EXPECT_EQ(expected_link_options, current_link_options);
}

TEST(Link_Compile_Options, compileOptionsTest_empty) {
    expected_link_options.clear();
    expected_compile_options.clear();
    std::string expected_compile_options = "";
    std::string expected_link_options = "";
    static sycl::unittest::PiImage DevImage = generateEAMTestKernelImage(expected_compile_options, expected_link_options);
    auto BundleObj = sycl::compile(DevImage);
    EXPECT_EQ(expected_link_options, current_link_options);
}

TEST(Link_Compile_Options, compileOptionsTest_one_param) {
    expected_link_options.clear();
    expected_compile_options.clear();
    std::string expected_compile_options = "-foo";
    std::string expected_link_options = "";
    static sycl::unittest::PiImage DevImage = generateEAMTestKernelImage(expected_compile_options, expected_link_options);
    auto BundleObj = sycl::compile(DevImage);
    EXPECT_EQ(expected_link_options, current_link_options);
}
