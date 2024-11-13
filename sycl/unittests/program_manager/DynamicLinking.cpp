#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/RuntimeLinkingCommon.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

namespace DynamicLinkingTest {
class BasicCaseKernel;
class UnresolvedDepKernel;
class MutualDepKernelA;
class MutualDepKernelB;
class AOTCaseKernel;
} // namespace DynamicLinkingTest

namespace sycl {
inline namespace _V1 {
namespace detail {
#define KERNEL_INFO(KernelName)                                                \
  template <>                                                                  \
  struct KernelInfo<DynamicLinkingTest::KernelName>                            \
      : public unittest::MockKernelInfoBase {                                  \
    static constexpr const char *getName() { return #KernelName; }             \
  };

KERNEL_INFO(BasicCaseKernel)
KERNEL_INFO(UnresolvedDepKernel)
KERNEL_INFO(MutualDepKernelA)
KERNEL_INFO(MutualDepKernelB)
KERNEL_INFO(AOTCaseKernel)

#undef KERNEL_INFO
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace {
std::vector<sycl::unittest::MockProperty>
createPropertySet(const std::vector<std::string> &Symbols) {
  sycl::unittest::MockPropertySet PropSet;
  std::vector<sycl::unittest::MockProperty> Props;
  for (const std::string &Symbol : Symbols) {
    std::vector<char> Storage(sizeof(uint32_t));
    uint32_t Val = 1;
    auto *DataPtr = reinterpret_cast<char *>(&Val);
    std::uninitialized_copy(DataPtr, DataPtr + sizeof(uint32_t),
                            Storage.data());

    sycl::unittest::MockProperty Prop(Symbol, Storage,
                                      SYCL_PROPERTY_TYPE_UINT32);

    Props.push_back(Prop);
  }
  return Props;
}

sycl::unittest::MockDeviceImage generateImage(
    std::initializer_list<std::string> KernelNames,
    const std::vector<std::string> &ExportedSymbols,
    const std::vector<std::string> &ImportedSymbols, unsigned char Magic,
    sycl::detail::ur::DeviceBinaryType BinType = SYCL_DEVICE_BINARY_TYPE_SPIRV,
    const char *DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_SPIRV64) {
  sycl::unittest::MockPropertySet PropSet;
  if (!ExportedSymbols.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                   createPropertySet(ExportedSymbols));
  if (!ImportedSymbols.empty())
    PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                   createPropertySet(ImportedSymbols));
  std::vector<unsigned char> Bin{Magic};

  std::vector<sycl::unittest::MockOffloadEntry> Entries =
      sycl::unittest::makeEmptyKernels(KernelNames);

  sycl::unittest::MockDeviceImage Img{BinType,
                                      DeviceTargetSpec,
                                      "", // Compile options
                                      "", // Link options
                                      std::move(Bin),
                                      std::move(Entries),
                                      std::move(PropSet)};

  return Img;
}

static constexpr unsigned BASIC_CASE_PRG = 2;
static constexpr unsigned BASIC_CASE_PRG_DEP = 3;
static constexpr unsigned BASIC_CASE_PRG_DEP_NATIVE = 5;
static constexpr unsigned BASIC_CASE_PRG_DEP_DEP = 7;
static constexpr unsigned UNRESOLVED_DEP_PRG = 11;
static constexpr unsigned MUTUAL_DEP_PRG_A = 13;
static constexpr unsigned MUTUAL_DEP_PRG_B = 17;
static constexpr unsigned AOT_CASE_PRG_NATIVE = 23;
static constexpr unsigned AOT_CASE_PRG_DEP_NATIVE = 29;

static sycl::unittest::MockDeviceImage Imgs[] = {
    generateImage({"BasicCaseKernel"}, {}, {"BasicCaseKernelDep"},
                  BASIC_CASE_PRG),
    generateImage({"BasicCaseKernelDep"}, {"BasicCaseKernelDep"},
                  {"BasicCaseKernelDepDep"}, BASIC_CASE_PRG_DEP),
    generateImage({"BasicCaseKernelDep"}, {"BasicCaseKernelDep"},
                  {"BasicCaseKernelDepDep"}, BASIC_CASE_PRG_DEP_NATIVE,
                  SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN),
    generateImage({"BasicCaseKernelDepDep"}, {"BasicCaseKernelDepDep"}, {},
                  BASIC_CASE_PRG_DEP_DEP),
    generateImage({"UnresolvedDepKernel"}, {},
                  {"UnresolvedDepKernelUnresolvedDep"}, UNRESOLVED_DEP_PRG),
    generateImage({"MutualDepKernelA", "MutualDepKernelBDep"},
                  {"MutualDepKernelBDep"}, {"MutualDepKernelADep"},
                  MUTUAL_DEP_PRG_A),
    generateImage({"MutualDepKernelADep", "MutualDepKernelB"},
                  {"MutualDepKernelADep"}, {"MutualDepKernelBDep"},
                  MUTUAL_DEP_PRG_B),
    generateImage({"AOTCaseKernel"}, {}, {"AOTCaseKernelDep"},
                  AOT_CASE_PRG_NATIVE, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN),
    generateImage({"AOTCaseKernelDep"}, {"AOTCaseKernelDep"}, {},
                  AOT_CASE_PRG_DEP_NATIVE, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN)};

// Registers mock devices images in the SYCL RT
static sycl::unittest::MockDeviceImageArray<9> ImgArray{Imgs};

TEST(DynamicLinking, BasicCase) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::BasicCaseKernel>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramCreateCalls, 3u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {BASIC_CASE_PRG, BASIC_CASE_PRG_DEP, BASIC_CASE_PRG_DEP_DEP}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            BASIC_CASE_PRG * BASIC_CASE_PRG_DEP * BASIC_CASE_PRG_DEP_DEP);
}

TEST(DynamicLinking, UnresolvedDep) {
  try {
    sycl::unittest::UrMock<> Mock;
    sycl::platform Plt = sycl::platform();
    sycl::queue Q(Plt.get_devices()[0]);
    Q.single_task<DynamicLinkingTest::UnresolvedDepKernel>([=]() {});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::build);
    EXPECT_STREQ(e.what(), "No device image found for external symbol "
                           "UnresolvedDepKernelUnresolvedDep");
  }
}

TEST(DynamicLinking, MutualDependency) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::MutualDepKernelA>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramCreateCalls, 2u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {MUTUAL_DEP_PRG_A, MUTUAL_DEP_PRG_B}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            MUTUAL_DEP_PRG_A * MUTUAL_DEP_PRG_B);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::MutualDepKernelB>([=]() {});
  // The program contianing this kernel should be taken from the
  // in-memory cache.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramCreateCalls, 0u);
}

TEST(DynamicLinking, AheadOfTime) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::AOTCaseKernel>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramCreateWithBinaryCalls, 2u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {AOT_CASE_PRG_NATIVE, AOT_CASE_PRG_DEP_NATIVE}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            AOT_CASE_PRG_NATIVE * AOT_CASE_PRG_DEP_NATIVE);
}

TEST(DynamicLinking, AheadOfTimeUnsupported) {
  try {
    sycl::unittest::UrMock<sycl::backend::ext_oneapi_level_zero> Mock;
    sycl::platform Plt = sycl::platform();
    sycl::queue Q(Plt.get_devices()[0]);
    Q.single_task<DynamicLinkingTest::AOTCaseKernel>([=]() {});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(), "Cannot resolve external symbols, linking is "
                           "unsupported for the backend");
  }
}

static ur_result_t redefined_urProgramCompileExp(void *pParams) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

TEST(DynamicLinking, UnsupportedCompileExp) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();
  mock::getCallbacks().set_replace_callback("urProgramCompileExp",
                                            redefined_urProgramCompileExp);

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::BasicCaseKernel>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramCreateCalls, 3u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {BASIC_CASE_PRG, BASIC_CASE_PRG_DEP, BASIC_CASE_PRG_DEP_DEP}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            BASIC_CASE_PRG * BASIC_CASE_PRG_DEP * BASIC_CASE_PRG_DEP_DEP);
}

} // anonymous namespace
