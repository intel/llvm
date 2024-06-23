#include <sycl/sycl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

namespace VirtualFunctionsTest {

class KernelA;
class KernelB;

} // namespace VirtualFunctionsTest

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<VirtualFunctionsTest::KernelA>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "KernelA"; }
};
template <>
struct KernelInfo<VirtualFunctionsTest::KernelB>
    : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return "KernelB"; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::PiImage generateImage(const std::string &KernelName,
                                             unsigned char Magic) {

  sycl::unittest::PiPropertySet PropSet;
  sycl::unittest::PiArray<sycl::unittest::PiProperty> Props;
  std::string VFSet = "set-a";
  uint64_t PropSize = VFSet.size();
  std::vector<char> Storage(/* bytes for size */ 8 + PropSize);
  std::uninitialized_copy(&PropSize, &PropSize + sizeof(uint64_t),
                          Storage.data());
  std::uninitialized_copy(VFSet.data(), VFSet.data() + PropSize,
                          Storage.data() + /* bytes for size */ 8);
  sycl::unittest::PiProperty Prop("uses-virtual-functions-set", Storage,
                                  PI_PROPERTY_TYPE_BYTE_ARRAY);
  Props.push_back(Prop);
  PropSet.insert(__SYCL_PI_PROPERTY_SET_SYCL_VIRTUAL_FUNCTIONS,
                 std::move(Props));

  std::vector<unsigned char> Bin{Magic};

  sycl::unittest::PiArray<sycl::unittest::PiOffloadEntry> Entries =
      sycl::unittest::makeEmptyKernels({KernelName});

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

static constexpr unsigned char PROGRAM_A = 42;
static constexpr unsigned char PROGRAM_B = 84;
static constexpr unsigned char PROGRAM_LINKED = 128;

static sycl::unittest::PiImage Imgs[] = {generateImage("KernelA", PROGRAM_A),
                                         generateImage("KernelB", PROGRAM_B)};

// Registers mock devices images in the SYCL RT
static sycl::unittest::PiImageArray<2> ImgArray{Imgs};

static unsigned NumOfPiProgramCreateCalls = 0;
static unsigned NumOfPiProgramLinkCalls = 0;
std::vector<unsigned char> LinkedPrograms;

static pi_result redefined_piProgramCreate(pi_context, const void *il,
                                           size_t length, pi_program *res) {
  auto *Magic = reinterpret_cast<const unsigned char *>(il);
  *res =
      createDummyHandleWithData<pi_program>(const_cast<unsigned char *>(Magic));
  ++NumOfPiProgramCreateCalls;
  return PI_SUCCESS;
}

static pi_result
redefined_piProgramLink(pi_context context, pi_uint32 num_devices,
                        const pi_device *device_list, const char *options,
                        pi_uint32 num_input_programs,
                        const pi_program *input_programs,
                        void (*pfn_notify)(pi_program program, void *user_data),
                        void *user_data, pi_program *ret_program) {
  for (pi_uint32 I = 0; I < num_input_programs; ++I)
    LinkedPrograms.push_back(
        *reinterpret_cast<DummyHandlePtrT>(input_programs[I])->MData);

  ++NumOfPiProgramLinkCalls;

  *ret_program = createDummyHandleWithData<pi_program>(
      const_cast<unsigned char *>(&PROGRAM_LINKED));
  return PI_SUCCESS;
}

TEST(VirtualFunctions, A) {
  sycl::unittest::PiMock Mock;

  Mock.redefine<sycl::detail::PiApiKind::piProgramCreate>(
      redefined_piProgramCreate);
  Mock.redefine<sycl::detail::PiApiKind::piProgramLink>(
      redefined_piProgramLink);

  sycl::platform Plt = Mock.getPlatform();
  sycl::queue Q(Plt.get_devices()[0]);

  NumOfPiProgramCreateCalls = 0;
  NumOfPiProgramLinkCalls = 0;
  LinkedPrograms.clear();

  // We need to make sure that when we submitted the first kernel, both device
  // images were linked together
  Q.single_task<VirtualFunctionsTest::KernelA>([=]() {});
  // We expect two programs to be created (one per each device image we have)
  ASSERT_EQ(NumOfPiProgramCreateCalls, 2u);
  // Both programs should be linked together
  ASSERT_EQ(NumOfPiProgramLinkCalls, 1u);
  ASSERT_TRUE(
      std::any_of(LinkedPrograms.begin(), LinkedPrograms.end(),
                  [=](unsigned char program) { return program == PROGRAM_A; }));
  ASSERT_TRUE(
      std::any_of(LinkedPrograms.begin(), LinkedPrograms.end(),
                  [=](unsigned char program) { return program == PROGRAM_B; }));

  NumOfPiProgramCreateCalls = 0;
  NumOfPiProgramLinkCalls = 0;

  // We need to make sure that when we submitted the second kernel, the same
  // linked device image from cache was used as for the previous kernel
  Q.single_task<VirtualFunctionsTest::KernelB>([=]() {});

  // No new programs shoud be created, we must re-use an existing one (linked)
  // from in-memory cache
  ASSERT_EQ(NumOfPiProgramCreateCalls, 0u);
  ASSERT_EQ(NumOfPiProgramLinkCalls, 0u);
}
