#include <sycl/sycl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

namespace VirtualFunctionsTest {

class KernelA;
class KernelB;
class KernelC;
class KernelD;
class KernelE;
class KernelF;
class KernelG;

} // namespace VirtualFunctionsTest

namespace sycl {
inline namespace _V1 {
namespace detail {

#define KERNEL_INFO(KernelName)                                                \
  template <>                                                                  \
  struct KernelInfo<VirtualFunctionsTest::KernelName>                          \
      : public unittest::MockKernelInfoBase {                                  \
    static constexpr const char *getName() { return #KernelName; }             \
  };

KERNEL_INFO(KernelA)
KERNEL_INFO(KernelB)
KERNEL_INFO(KernelC)
KERNEL_INFO(KernelD)
KERNEL_INFO(KernelE)
KERNEL_INFO(KernelF)
KERNEL_INFO(KernelG)

#undef KERNEL_INFO

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::PiImage
generateImage(std::initializer_list<std::string> KernelNames,
              const std::string &VFSets, bool UsesVFSets, unsigned char Magic) {
  sycl::unittest::PiPropertySet PropSet;
  sycl::unittest::PiArray<sycl::unittest::PiProperty> Props;
  uint64_t PropSize = VFSets.size();
  std::vector<char> Storage(/* bytes for size */ 8 + PropSize +
                            /* null terminator */ 1);
  auto *SizePtr = reinterpret_cast<char *>(&PropSize);
  std::uninitialized_copy(SizePtr, SizePtr + sizeof(uint64_t), Storage.data());
  std::uninitialized_copy(VFSets.data(), VFSets.data() + PropSize,
                          Storage.data() + /* bytes for size */ 8);
  Storage.back() = '\0';
  const std::string PropName =
      UsesVFSets ? "uses-virtual-functions-set" : "virtual-functions-set";
  sycl::unittest::PiProperty Prop(PropName, Storage,
                                  PI_PROPERTY_TYPE_BYTE_ARRAY);

  Props.push_back(Prop);
  PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS,
                 std::move(Props));

  std::vector<unsigned char> Bin{Magic};

  sycl::unittest::PiArray<sycl::unittest::PiOffloadEntry> Entries =
      sycl::unittest::makeEmptyKernels(KernelNames);

  sycl::unittest::PiImage Img{
      PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
      __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
      "",                                     // Compile options
      "",                                     // Link options
      std::move(Bin),
      std::move(Entries),
      std::move(PropSet)};

  return Img;
}

static constexpr unsigned PROGRAM_A = 3;
static constexpr unsigned PROGRAM_A0 = 5;
static constexpr unsigned PROGRAM_B = 7;
static constexpr unsigned PROGRAM_B0 = 11;
static constexpr unsigned PROGRAM_B1 = 13;
static constexpr unsigned PROGRAM_C = 17;
static constexpr unsigned PROGRAM_C0 = 19;
static constexpr unsigned PROGRAM_C1 = 23;
static constexpr unsigned PROGRAM_D = 27;
static constexpr unsigned PROGRAM_D0 = 29;
static constexpr unsigned PROGRAM_E = 31;
static constexpr unsigned PROGRAM_E0 = 37;
static constexpr unsigned PROGRAM_F = 41;
static constexpr unsigned PROGRAM_F0 = 47;
static constexpr unsigned PROGRAM_F1 = 53;

// Device images with no entires are ignored by SYCL RT during registration.
// Therefore, we have to provide some kernel names to make the test work, even
// if we don't really have them/use them.
static sycl::unittest::PiImage Imgs[] = {
    generateImage({"KernelA"}, "set-a", /* uses vf set */ true, PROGRAM_A),
    generateImage({"DummyKernel0"}, "set-a", /* provides vf set */ false,
                  PROGRAM_A0),
    generateImage({"KernelB"}, "set-b", /* uses vf set */ true, PROGRAM_B),
    generateImage({"DummyKernel1"}, "set-b", /* provides vf set */ false,
                  PROGRAM_B0),
    generateImage({"DummyKernel2"}, "set-b", /* provides vf set */ false,
                  PROGRAM_B1),
    generateImage({"KernelC"}, "set-c1,set-c2", /* uses vf set */ true,
                  PROGRAM_C),
    generateImage({"DummyKernel3"}, "set-c1", /* provides vf set */ false,
                  PROGRAM_C0),
    generateImage({"DummyKernel4"}, "set-c2", /* provides vf set */ false,
                  PROGRAM_C1),
    generateImage({"KernelD"}, "set-d", /* uses vf set */ true, PROGRAM_D),
    generateImage({"DummyKernel5"}, "set-d", /* provides vf set */ false,
                  PROGRAM_D0),
    generateImage({"KernelE"}, "set-e,set-d", /* uses vf set */ true,
                  PROGRAM_E),
    generateImage({"DummyKernel6"}, "set-e", /* provides vf set */ false,
                  PROGRAM_E0),
    generateImage({"KernelF"}, "set-f", /* uses vf set */ true, PROGRAM_F),
    generateImage({"DummyKernel7"}, "set-f", /* provides vf set */ false,
                  PROGRAM_F0),
    generateImage({"KernelG"}, "set-f", /* uses vf set */ true, PROGRAM_F1)};

// Registers mock devices images in the SYCL RT
static sycl::unittest::PiImageArray<15> ImgArray{Imgs};

// Helper holder for all the data we want to capture from mocked APIs
struct CapturesHolder {
  unsigned NumOfPiProgramCreateCalls = 0;
  unsigned NumOfPiProgramLinkCalls = 0;
  unsigned ProgramUsedToCreateKernel = 0;
  std::vector<unsigned> LinkedPrograms;

  bool LinkedProgramsContains(std::initializer_list<unsigned> Programs) {
    return std::all_of(Programs.begin(), Programs.end(), [this](unsigned Prg) {
      return std::any_of(
          LinkedPrograms.begin(), LinkedPrograms.end(),
          [Prg](unsigned LinkedPrg) { return LinkedPrg == Prg; });
    });
  }

  void clear() {
    NumOfPiProgramCreateCalls = 0;
    NumOfPiProgramLinkCalls = 0;
    ProgramUsedToCreateKernel = 0;
    LinkedPrograms.clear();
  }
};

static CapturesHolder CapturedData;

static pi_result redefined_piProgramCreate(pi_context, const void *il,
                                           size_t length, pi_program *res) {
  auto *Magic = reinterpret_cast<const unsigned char *>(il);
  *res = createDummyHandle<pi_program>(sizeof(unsigned));
  reinterpret_cast<DummyHandlePtrT>(*res)->setDataAs<unsigned>(*Magic);
  ++CapturedData.NumOfPiProgramCreateCalls;
  return PI_SUCCESS;
}

static pi_result
redefined_piProgramLink(pi_context context, pi_uint32 num_devices,
                        const pi_device *device_list, const char *options,
                        pi_uint32 num_input_programs,
                        const pi_program *input_programs,
                        void (*pfn_notify)(pi_program program, void *user_data),
                        void *user_data, pi_program *ret_program) {
  unsigned ResProgram = 1;
  for (pi_uint32 I = 0; I < num_input_programs; ++I) {
    auto Val = reinterpret_cast<DummyHandlePtrT>(input_programs[I])
                   ->getDataAs<unsigned>();
    ResProgram *= Val;
    CapturedData.LinkedPrograms.push_back(Val);
  }

  ++CapturedData.NumOfPiProgramLinkCalls;

  *ret_program = createDummyHandle<pi_program>(sizeof(unsigned));
  reinterpret_cast<DummyHandlePtrT>(*ret_program)
      ->setDataAs<unsigned>(ResProgram);
  return PI_SUCCESS;
}

static pi_result redefined_piKernelCreate(pi_program program,
                                          const char *kernel_name,
                                          pi_kernel *ret_kernel) {
  CapturedData.ProgramUsedToCreateKernel =
      reinterpret_cast<DummyHandlePtrT>(program)->getDataAs<unsigned>();
  *ret_kernel = createDummyHandle<pi_kernel>();
  return PI_SUCCESS;
}

static sycl::unittest::PiMock setupMock() {
  sycl::unittest::PiMock Mock;

  Mock.redefine<sycl::detail::PiApiKind::piProgramCreate>(
      redefined_piProgramCreate);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(
      redefined_piProgramLink);
  Mock.redefine<sycl::detail::PiApiKind::piKernelCreate>(
      redefined_piKernelCreate);

  return Mock;
}

TEST(VirtualFunctions, SingleKernelUsesSingleVFSet) {
  auto Mock = setupMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedData.clear();

  // KernelA uses set "set-a" of virtual functions.
  Q.single_task<VirtualFunctionsTest::KernelA>([=]() {});
  // When we submit this kernel, we expect that two programs were created (one
  // for a kernel and another providing virtual functions set for it).
  ASSERT_EQ(CapturedData.NumOfPiProgramCreateCalls, 2u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedData.LinkedProgramsContains({PROGRAM_A, PROGRAM_A0}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedData.ProgramUsedToCreateKernel, PROGRAM_A * PROGRAM_A0);
}

TEST(VirtualFunctions, SingleKernelUsesSingleVFSetProvidedTwice) {
  auto Mock = setupMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedData.clear();

  // KernelB uses set "set-b" of virtual functions that is provided by two
  // device images.
  Q.single_task<VirtualFunctionsTest::KernelB>([=]() {});
  // When we submit this kernel, we expect that three programs were created (one
  // for a kernel and another two providing virtual functions set for it).
  ASSERT_EQ(CapturedData.NumOfPiProgramCreateCalls, 3u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(
      CapturedData.LinkedProgramsContains({PROGRAM_B, PROGRAM_B0, PROGRAM_B1}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedData.ProgramUsedToCreateKernel,
            PROGRAM_B * PROGRAM_B0 * PROGRAM_B1);
}

TEST(VirtualFunctions, SingleKernelUsesDifferentVFSets) {
  auto Mock = setupMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedData.clear();

  // KernelC uses set "set-c1" and "set-c2" of virtual functions which are
  // provided by two device images.
  Q.single_task<VirtualFunctionsTest::KernelC>([=]() {});
  // When we submit this kernel, we expect that three programs were created (one
  // for a kernel and another two providing virtual functions set for it).
  ASSERT_EQ(CapturedData.NumOfPiProgramCreateCalls, 3u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(
      CapturedData.LinkedProgramsContains({PROGRAM_C, PROGRAM_C0, PROGRAM_C1}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedData.ProgramUsedToCreateKernel,
            PROGRAM_C * PROGRAM_C0 * PROGRAM_C1);
}

TEST(VirtualFunctions, RecursiveSearchOfDependentDeviceImages) {
  auto Mock = setupMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedData.clear();

  // KernelD uses set "set-e" and "set-d" of virtual functions that is provided
  // by two device images. Additionally, "set-e" is also used by "KernelE"
  Q.single_task<VirtualFunctionsTest::KernelD>([=]() {});
  // When we submit this kernel, we expect that four programs were created (one
  // for KernelD  and another providing "set-d", as well as one for KernelE and
  // another providing "set-e").
  ASSERT_EQ(CapturedData.NumOfPiProgramCreateCalls, 4u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(CapturedData.LinkedProgramsContains(
      {PROGRAM_D, PROGRAM_D0, PROGRAM_E, PROGRAM_E0}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedData.ProgramUsedToCreateKernel,
            PROGRAM_D * PROGRAM_D0 * PROGRAM_E * PROGRAM_E0);
}

TEST(VirtualFunctions, TwoKernelsShareTheSameSet) {
  auto Mock = setupMock();

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  CapturedData.clear();

  // KernelF uses set "set-f" that is also used by KernelG
  Q.single_task<VirtualFunctionsTest::KernelF>([=]() {});
  // When we submit this kernel, we expect that three programs were created (one
  // for KernelF, another providing "set-f" and one more for KernelG)
  ASSERT_EQ(CapturedData.NumOfPiProgramCreateCalls, 3u);
  // Both programs should be linked together.
  ASSERT_EQ(CapturedData.NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(
      CapturedData.LinkedProgramsContains({PROGRAM_F, PROGRAM_F0, PROGRAM_F1}));
  // And the linked program should be used to create a kernel.
  ASSERT_EQ(CapturedData.ProgramUsedToCreateKernel,
            PROGRAM_F * PROGRAM_F0 * PROGRAM_F1);

  CapturedData.clear();

  // When we submit a second kernel, we expect that no new programs will be
  // created and we will simply use previously linked program for that kernel.
  Q.single_task<VirtualFunctionsTest::KernelG>([=]() {});
  ASSERT_EQ(CapturedData.NumOfPiProgramCreateCalls, 0u);
  ASSERT_EQ(CapturedData.NumOfPiProgramLinkCalls, 0u);
  ASSERT_EQ(CapturedData.ProgramUsedToCreateKernel,
            PROGRAM_F * PROGRAM_F0 * PROGRAM_F1);
}

// TODO: Add test cases for kernel_bundle usage
