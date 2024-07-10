#include <sycl/sycl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>
#include <helpers/RuntimeLinkingCommon.hpp>

#include <gtest/gtest.h>

namespace DynamicLinkingTest {
class KernelA;
class KernelB;
class KernelC;
class KernelD;
class KernelE;
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

KERNEL_INFO(KernelA)
KERNEL_INFO(KernelB)
KERNEL_INFO(KernelC)
KERNEL_INFO(KernelD)
KERNEL_INFO(KernelE)

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
                   std::move(createPropertySet(ExportedSymbols)));
  if (!ImportedSymbols.empty())
    PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                   std::move(createPropertySet(ImportedSymbols)));
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

static constexpr unsigned PROGRAM_A = 2;
static constexpr unsigned PROGRAM_A_DEP = 3;
static constexpr unsigned PROGRAM_A_DEP_NATIVE = 5;
static constexpr unsigned PROGRAM_A_DEP_DEP = 7;
static constexpr unsigned PROGRAM_B = 11;
static constexpr unsigned PROGRAM_C = 13;
static constexpr unsigned PROGRAM_D = 17;
static constexpr unsigned PROGRAM_E_NATIVE = 23;
static constexpr unsigned PROGRAM_E_DEP_NATIVE = 29;

static sycl::unittest::PiImage Imgs[] = {
    generateImage({"KernelA"}, {}, {"KernelADep"}, PROGRAM_A),
    generateImage({"KernelADep"}, {"KernelADep"}, {"KernelADepDep"},
                  PROGRAM_A_DEP),
    generateImage({"KernelADep"}, {"KernelADep"}, {"KernelADepDep"},
                  PROGRAM_A_DEP_NATIVE, PI_DEVICE_BINARY_TYPE_NATIVE),
    generateImage({"KernelADepDep"}, {"KernelADepDep"}, {}, PROGRAM_A_DEP_DEP),
    generateImage({"KernelB"}, {}, {"KernelBUnresolvedDep"}, PROGRAM_B),
    generateImage({"KernelC", "KernelDDep"}, {"KernelDDep"}, {"KernelCDep"},
                  PROGRAM_C),
    generateImage({"KernelCDep", "KernelD"}, {"KernelCDep"}, {"KernelDDep"},
                  PROGRAM_D),
    generateImage({"KernelE"}, {}, {"KernelEDep"}, PROGRAM_E_NATIVE,
                  PI_DEVICE_BINARY_TYPE_NATIVE),
    generateImage({"KernelEDep"}, {"KernelEDep"}, {}, PROGRAM_E_DEP_NATIVE,
                  PI_DEVICE_BINARY_TYPE_NATIVE)};

// Registers mock devices images in the SYCL RT
static sycl::unittest::PiImageArray<9> ImgArray{Imgs};

TEST(DynamicLinking, BasicCase) {
  auto Mock = setupRuntimeLinkingMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::KernelA>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramCreateCalls, 3u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {PROGRAM_A, PROGRAM_A_DEP, PROGRAM_A_DEP_DEP}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            PROGRAM_A * PROGRAM_A_DEP * PROGRAM_A_DEP_DEP);
}

TEST(DynamicLinking, UnresolvedDep) {
  try {
    sycl::unittest::PiMock Mock;
    sycl::platform Plt = Mock.getPlatform();
    sycl::queue Q(Plt.get_devices()[0]);
    Q.single_task<DynamicLinkingTest::KernelB>([=]() {});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::build);
    EXPECT_STREQ(
        e.what(),
        "No device image found for external symbol KernelBUnresolvedDep");
  }
}

TEST(DynamicLinking, MutualDependency) {
  auto Mock = setupRuntimeLinkingMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::KernelC>([=]() {});
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramCreateCalls, 2u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(
      CapturedLinkingData.LinkedProgramsContains({PROGRAM_C, PROGRAM_D}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            PROGRAM_C * PROGRAM_D);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::KernelD>([=]() {});
  // The program contianing this kernel should be taken from the
  // in-memory cache.
  ASSERT_EQ(CapturedLinkingData.NumOfPiProgramCreateCalls, 0u);
}

TEST(DynamicLinking, AheadOfTime) {
  try {
    sycl::unittest::PiMock Mock;
    sycl::platform Plt = Mock.getPlatform();
    sycl::queue Q(Plt.get_devices()[0]);
    Q.single_task<DynamicLinkingTest::KernelE>([=]() {});
    FAIL();
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::feature_not_supported);
    EXPECT_STREQ(e.what(),
                 "Dynamic linking is not supported for AOT compilation yet");
  }
}

} // anonymous namespace