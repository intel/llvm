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

const static sycl::specialization_id<int> SpecConst1{1};
const static sycl::specialization_id<int> SpecConst2{2};

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

template <> const char *get_spec_constant_symbolic_ID<SpecConst1>() {
  return "SC1";
}
template <> const char *get_spec_constant_symbolic_ID<SpecConst2>() {
  return "SC2";
}

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

sycl::unittest::MockDeviceImage
generateImage(std::initializer_list<std::string> KernelNames,
              const std::vector<std::string> &ExportedSymbols,
              const std::vector<std::string> &ImportedSymbols,
              unsigned char Magic, sycl::detail::ur::DeviceBinaryType BinType,
              const char *DeviceTargetSpec,
              sycl::unittest::MockPropertySet PropSet) {
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

sycl::unittest::MockDeviceImage generateImage(
    std::initializer_list<std::string> KernelNames,
    const std::vector<std::string> &ExportedSymbols,
    const std::vector<std::string> &ImportedSymbols, unsigned char Magic,
    const std::string &SCName, uint32_t SCID, int SCValue,
    sycl::detail::ur::DeviceBinaryType BinType = SYCL_DEVICE_BINARY_TYPE_SPIRV,
    const char *DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_SPIRV64) {
  sycl::unittest::MockPropertySet PropSet;
  std::vector<char> SpecConstData;
  sycl::unittest::MockProperty SC = sycl::unittest::makeSpecConstant<int>(
      SpecConstData, SCName, {SCID}, {0}, {SCValue});
  sycl::unittest::addSpecConstants({SC}, std::move(SpecConstData), PropSet);
  return generateImage(KernelNames, ExportedSymbols, ImportedSymbols, Magic,
                       SYCL_DEVICE_BINARY_TYPE_SPIRV,
                       __SYCL_DEVICE_BINARY_TARGET_SPIRV64, PropSet);
}
sycl::unittest::MockDeviceImage generateImage(
    std::initializer_list<std::string> KernelNames,
    const std::vector<std::string> &ExportedSymbols,
    const std::vector<std::string> &ImportedSymbols, unsigned char Magic,
    sycl::detail::ur::DeviceBinaryType BinType = SYCL_DEVICE_BINARY_TYPE_SPIRV,
    const char *DeviceTargetSpec = __SYCL_DEVICE_BINARY_TARGET_SPIRV64) {
  return generateImage(KernelNames, ExportedSymbols, ImportedSymbols, Magic,
                       BinType, DeviceTargetSpec, {});
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
                  MUTUAL_DEP_PRG_A, "SC1", 0, 1),
    generateImage({"MutualDepKernelADep", "MutualDepKernelB"},
                  {"MutualDepKernelADep"}, {"MutualDepKernelBDep"},
                  MUTUAL_DEP_PRG_B, "SC2", 1, 2),
    generateImage({"AOTCaseKernel"}, {}, {"AOTCaseKernelDep"},
                  AOT_CASE_PRG_NATIVE, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN),
    generateImage({"AOTCaseKernelDep"}, {"AOTCaseKernelDep"}, {},
                  AOT_CASE_PRG_DEP_NATIVE, SYCL_DEVICE_BINARY_TYPE_NATIVE,
                  __SYCL_DEVICE_BINARY_TARGET_SPIRV64_GEN)};

// Registers mock devices images in the SYCL RT
static sycl::unittest::MockDeviceImageArray<9> ImgArray{Imgs};

void runCommonBasicCaseChecks() {
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramCreateCalls, 3u);
  // The three programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {BASIC_CASE_PRG, BASIC_CASE_PRG_DEP, BASIC_CASE_PRG_DEP_DEP}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            BASIC_CASE_PRG * BASIC_CASE_PRG_DEP * BASIC_CASE_PRG_DEP_DEP);
}

TEST(DynamicLinking, BasicCase) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::BasicCaseKernel>([=]() {});

  runCommonBasicCaseChecks();
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

void runCommonMutualDepTestChecks() {
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramCreateCalls, 2u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {MUTUAL_DEP_PRG_A, MUTUAL_DEP_PRG_B}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            MUTUAL_DEP_PRG_A * MUTUAL_DEP_PRG_B);
}

TEST(DynamicLinking, MutualDependency) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  Q.single_task<DynamicLinkingTest::MutualDepKernelA>([=]() {});
  runCommonMutualDepTestChecks();

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
  // The three programs should be linked together.
  ASSERT_EQ(CapturedLinkingData.NumOfUrProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedLinkingData.LinkedProgramsContains(
      {BASIC_CASE_PRG, BASIC_CASE_PRG_DEP, BASIC_CASE_PRG_DEP_DEP}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedLinkingData.ProgramUsedToCreateKernel,
            BASIC_CASE_PRG * BASIC_CASE_PRG_DEP * BASIC_CASE_PRG_DEP_DEP);
}

template <typename KernelName>
void testKernelBundleBuild(
    std::size_t NKernelIDsExpected,
    const std::vector<sycl::kernel_id> &KernelIDsRequested = {
        sycl::get_kernel_id<KernelName>()}) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  sycl::kernel_bundle KB =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(
          Q.get_context(), KernelIDsRequested);
  // Only one linked resulting image expected
  EXPECT_EQ(std::distance(KB.begin(), KB.end()), 1);
  EXPECT_EQ(KB.get_kernel_ids().size(), NKernelIDsExpected);

  Q.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(KB);
    CGH.single_task<KernelName>([=]() {});
  });
}

TEST(DynamicLinking, KernelBundleBuild) {
  testKernelBundleBuild<DynamicLinkingTest::BasicCaseKernel>(
      /*NKernelIDsExpected*/ 1u);
  runCommonBasicCaseChecks();
}

template <typename KernelName>
void testKernelBundleCompileLink(
    long NImagesExpectedBeforeLink, std::size_t NKernelIDsExpected,
    const std::vector<sycl::kernel_id> &KernelIDsRequested = {
        sycl::get_kernel_id<KernelName>()}) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  sycl::kernel_bundle InputKB =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Q.get_context(),
                                                         KernelIDsRequested);
  EXPECT_EQ(std::distance(InputKB.begin(), InputKB.end()),
            NImagesExpectedBeforeLink);
  EXPECT_EQ(InputKB.get_kernel_ids().size(), NKernelIDsExpected);

  sycl::kernel_bundle ObjectKB = sycl::compile(InputKB);
  EXPECT_EQ(std::distance(ObjectKB.begin(), ObjectKB.end()),
            NImagesExpectedBeforeLink);
  EXPECT_EQ(ObjectKB.get_kernel_ids().size(), NKernelIDsExpected);

  sycl::kernel_bundle LinkedKB = sycl::link({ObjectKB});
  // Only one linked resulting image expected
  EXPECT_EQ(std::distance(LinkedKB.begin(), LinkedKB.end()), 1);
  EXPECT_EQ(LinkedKB.get_kernel_ids().size(), NKernelIDsExpected);

  Q.submit([&](sycl::handler &CGH) {
    CGH.use_kernel_bundle(LinkedKB);
    CGH.single_task<KernelName>([=]() {});
  });
}

TEST(DynamicLinking, KernelBundleCompileLink) {
  testKernelBundleCompileLink<DynamicLinkingTest::BasicCaseKernel>(
      /*NImagesExpectedBeforeLink*/ 3, /*NKernelIDsExpected*/ 1);
  runCommonBasicCaseChecks();
}

TEST(DynamicLinking, KernelBundleMutualDep) {
  testKernelBundleCompileLink<
      DynamicLinkingTest::
          MutualDepKernelA>(/*NImagesExpectedBeforeLink*/
                            2, /*NKernelIDsExpected*/ 2,
                            {sycl::get_kernel_id<
                                 DynamicLinkingTest::MutualDepKernelA>(),
                             sycl::get_kernel_id<
                                 DynamicLinkingTest::MutualDepKernelB>()});
  runCommonMutualDepTestChecks();
}

// Test that the dependency image and its kernel id are exposed as part of the
// kernel bundle even when not explicitly requested.
TEST(DynamicLinking, KernelBundleMutualDepCompileLinkIndirect) {
  testKernelBundleCompileLink<
      DynamicLinkingTest::
          MutualDepKernelA>(/*NImagesExpectedBeforeLink*/
                            2, /*NKernelIDsExpected*/ 2,
                            {sycl::get_kernel_id<
                                DynamicLinkingTest::MutualDepKernelB>()});
  runCommonMutualDepTestChecks();
}

TEST(DynamicLinking, KernelBundleMutualDepBuildIndirect) {
  testKernelBundleBuild<DynamicLinkingTest::MutualDepKernelB>(
      /*NKernelIDsExpected*/ 2u,
      {sycl::get_kernel_id<DynamicLinkingTest::MutualDepKernelA>()});
  runCommonMutualDepTestChecks();
}

TEST(DynamicLinking, KernelBundleSpecConstsCompileLink) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  sycl::kernel_bundle InputKB =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          Q.get_context(),
          {sycl::get_kernel_id<DynamicLinkingTest::MutualDepKernelA>()});
  EXPECT_EQ(InputKB.get_specialization_constant<SpecConst1>(), 1);
  EXPECT_EQ(InputKB.get_specialization_constant<SpecConst2>(), 2);

  InputKB.set_specialization_constant<SpecConst1>(10);
  InputKB.set_specialization_constant<SpecConst2>(20);
  sycl::kernel_bundle ObjectKB = sycl::compile(InputKB);
  EXPECT_EQ(ObjectKB.get_specialization_constant<SpecConst1>(), 10);
  EXPECT_EQ(ObjectKB.get_specialization_constant<SpecConst2>(), 20);

  sycl::kernel_bundle LinkedKB = sycl::link({ObjectKB});
  EXPECT_EQ(LinkedKB.get_specialization_constant<SpecConst1>(), 10);
  EXPECT_EQ(LinkedKB.get_specialization_constant<SpecConst2>(), 20);
}

TEST(DynamicLinking, KernelBundleSpecConstsBuild) {
  sycl::unittest::UrMock<> Mock;
  setupRuntimeLinkingMock();

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedLinkingData.clear();

  sycl::kernel_bundle InputKB =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          Q.get_context(),
          {sycl::get_kernel_id<DynamicLinkingTest::MutualDepKernelB>()});
  EXPECT_EQ(InputKB.get_specialization_constant<SpecConst1>(), 1);
  EXPECT_EQ(InputKB.get_specialization_constant<SpecConst2>(), 2);

  InputKB.set_specialization_constant<SpecConst1>(10);
  InputKB.set_specialization_constant<SpecConst2>(20);
  sycl::kernel_bundle BuiltKB = sycl::build(InputKB);
  EXPECT_EQ(BuiltKB.get_specialization_constant<SpecConst1>(), 10);
  EXPECT_EQ(BuiltKB.get_specialization_constant<SpecConst2>(), 20);
}
} // anonymous namespace
