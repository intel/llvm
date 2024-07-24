#include <sycl/sycl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/RuntimeLinkingCommon.hpp>

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
sycl::unittest::PiArray<sycl::unittest::PiProperty>
createPropertySet(const std::vector<std::string> &Symbols) {
  sycl::unittest::PiPropertySet PropSet;
  sycl::unittest::PiArray<sycl::unittest::PiProperty> Props;
  for (const std::string &Symbol : Symbols) {
    std::vector<char> Storage(sizeof(pi_uint32));
    uint32_t Val = 1;
    auto *DataPtr = reinterpret_cast<char *>(&Val);
    std::uninitialized_copy(DataPtr, DataPtr + sizeof(uint32_t),
                            Storage.data());
    sycl::unittest::PiProperty Prop(Symbol, Storage, PI_PROPERTY_TYPE_UINT32);

    Props.push_back(Prop);
  }
  return Props;
}

sycl::unittest::PiImage
generateImage(std::initializer_list<std::string> KernelNames,
              const std::vector<std::string> &ExportedSymbols,
              const std::vector<std::string> &ImportedSymbols,
              unsigned char Magic,
              sycl::detail::pi::PiDeviceBinaryType BinType =
                  PI_DEVICE_BINARY_TYPE_SPIRV) {
  sycl::unittest::PiPropertySet PropSet;
  if (!ExportedSymbols.empty())
    PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                   createPropertySet(ExportedSymbols));
  if (!ImportedSymbols.empty())
    PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                   createPropertySet(ImportedSymbols));
  std::vector<unsigned char> Bin{Magic};

  sycl::unittest::PiArray<sycl::unittest::PiOffloadEntry> Entries =
      sycl::unittest::makeEmptyKernels(KernelNames);

  sycl::unittest::PiImage Img{
      BinType,
      __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
      "",                                     // Compile options
      "",                                     // Link options
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

static sycl::unittest::PiImage Imgs[] = {
    generateImage({"BasicCaseKernel"}, {}, {"BasicCaseKernelDep"},
                  BASIC_CASE_PRG),
    generateImage({"BasicCaseKernelDep"}, {"BasicCaseKernelDep"},
                  {"BasicCaseKernelDepDep"}, BASIC_CASE_PRG_DEP),
    generateImage({"BasicCaseKernelDep"}, {"BasicCaseKernelDep"},
                  {"BasicCaseKernelDepDep"}, BASIC_CASE_PRG_DEP_NATIVE,
                  PI_DEVICE_BINARY_TYPE_NATIVE),
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
                  AOT_CASE_PRG_NATIVE, PI_DEVICE_BINARY_TYPE_NATIVE),
    generateImage({"AOTCaseKernelDep"}, {"AOTCaseKernelDep"}, {},
                  AOT_CASE_PRG_DEP_NATIVE, PI_DEVICE_BINARY_TYPE_NATIVE)};

// Registers mock devices images in the SYCL RT
static sycl::unittest::PiImageArray<9> ImgArray{Imgs};

TEST(DynamicLinking, BasicCase) {
  auto Mock = setupRuntimeLinkingMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::BasicCaseKernel>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramCreateCalls, 3u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {BASIC_CASE_PRG, BASIC_CASE_PRG_DEP, BASIC_CASE_PRG_DEP_DEP}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            BASIC_CASE_PRG * BASIC_CASE_PRG_DEP * BASIC_CASE_PRG_DEP_DEP);
}

TEST(DynamicLinking, UnresolvedDep) {
  try {
    sycl::unittest::PiMock Mock;
    sycl::platform Plt = Mock.getPlatform();
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
  auto Mock = setupRuntimeLinkingMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::MutualDepKernelA>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramCreateCalls, 2u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {MUTUAL_DEP_PRG_A, MUTUAL_DEP_PRG_B}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            MUTUAL_DEP_PRG_A * MUTUAL_DEP_PRG_B);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::MutualDepKernelB>([=]() {});
  // The program contianing this kernel should be taken from the
  // in-memory cache.
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramCreateCalls, 0u);
}

TEST(DynamicLinking, AheadOfTime) {
  try {
    sycl::unittest::PiMock Mock;
    sycl::platform Plt = Mock.getPlatform();
    sycl::queue Q(Plt.get_devices()[0]);
    Q.single_task<DynamicLinkingTest::AOTCaseKernel>([=]() {});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(),
                 "Dynamic linking is not supported for AOT compilation yet");
  }
}

} // anonymous namespace
