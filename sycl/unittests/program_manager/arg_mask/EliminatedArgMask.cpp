//==------- EliminatedArgMask.cpp --- eliminated args mask unit test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/config.hpp>
#include <detail/handler_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

class EAMTestKernel;
class EAMTestKernel2;
class EAMTestKernel3;
constexpr const char EAMTestKernelName[] = "EAMTestKernel";
constexpr const char EAMTestKernel2Name[] = "EAMTestKernel2";
constexpr const char EAMTestKernel3Name[] = "EAMTestKernel3";
constexpr unsigned EAMTestKernelNumArgs = 4;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<EAMTestKernel> : public unittest::MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs; }
  static constexpr const char *getName() { return EAMTestKernelName; }
};

template <>
struct KernelInfo<EAMTestKernel2> : public unittest::MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return 0; }
  static constexpr const char *getName() { return EAMTestKernel2Name; }
};

template <>
struct KernelInfo<EAMTestKernel3> : public unittest::MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return EAMTestKernelNumArgs; }
  static constexpr const char *getName() { return EAMTestKernel3Name; }
};

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::MockDeviceImage generateEAMTestKernelImage() {
  using namespace sycl::unittest;

  // Eliminated arguments are 1st and 3rd.
  std::vector<unsigned char> KernelEAM{0b00000101};
  MockProperty EAMKernelPOI = makeKernelParamOptInfo(
      EAMTestKernelName, EAMTestKernelNumArgs, KernelEAM);
  std::vector<MockProperty> ImgKPOI{std::move(EAMKernelPOI)};

  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO, std::move(ImgKPOI));

  std::vector<MockOffloadEntry> Entries = makeEmptyKernels({EAMTestKernelName});

  MockDeviceImage Img{std::move(Entries), std::move(PropSet)};

  return Img;
}

static sycl::unittest::MockDeviceImage generateEAMTestKernel3Image() {
  using namespace sycl::unittest;

  // Eliminated arguments are 2nd and 4th.
  std::vector<unsigned char> KernelEAM{0b00001010};
  MockProperty EAMKernelPOI = makeKernelParamOptInfo(
      EAMTestKernel3Name, EAMTestKernelNumArgs, KernelEAM);
  std::vector<MockProperty> ImgKPOI{std::move(EAMKernelPOI)};

  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_KERNEL_PARAM_OPT_INFO, std::move(ImgKPOI));

  std::vector<MockOffloadEntry> Entries =
      makeEmptyKernels({EAMTestKernel3Name});

  MockDeviceImage Img(std::move(Entries), std::move(PropSet));

  return Img;
}

static sycl::unittest::MockDeviceImage EAMImg = generateEAMTestKernelImage();
static sycl::unittest::MockDeviceImage EAM2Img =
    sycl::unittest::generateDefaultImage({EAMTestKernel2Name});
static sycl::unittest::MockDeviceImage EAM3Img = generateEAMTestKernel3Image();
static sycl::unittest::MockDeviceImageArray<1> EAMImgArray{&EAMImg};
static sycl::unittest::MockDeviceImageArray<1> EAM2ImgArray{&EAM2Img};
static sycl::unittest::MockDeviceImageArray<1> EAM3ImgArray{&EAM3Img};

// ur_program_handle_t address is used as a key for ProgramManager::NativePrograms
// storage. redefinedProgramLinkCommon makes ur_program_handle_t address equal to 0x1.
// Make sure that size of Bin is different for device images used in these tests
// and greater than 1.
inline ur_result_t redefinedProgramCreateEAM(void *pParams) {
  auto params = *static_cast<ur_program_create_with_il_params_t *>(pParams);
  static size_t UrProgramAddr = 2;
  **params.pphProgram = reinterpret_cast<ur_program_handle_t>(UrProgramAddr++);
  return UR_RESULT_SUCCESS;
}

class MockHandler : public sycl::handler {

public:
  using sycl::handler::impl;

  MockHandler(std::shared_ptr<sycl::detail::queue_impl> Queue)
      : sycl::handler(Queue, /*CallerNeedsEvent*/ true) {}

  std::unique_ptr<sycl::detail::CG> finalize() {
    auto CGH = static_cast<sycl::handler *>(this);
    std::unique_ptr<sycl::detail::CG> CommandGroup;
    switch (getType()) {
    case sycl::detail::CGType::Kernel: {
      CommandGroup.reset(new sycl::detail::CGExecKernel(
          std::move(impl->MNDRDesc), std::move(CGH->MHostKernel),
          std::move(CGH->MKernel), std::move(impl->MKernelBundle),
          std::move(impl->CGData), std::move(impl->MArgs),
          CGH->MKernelName.c_str(), std::move(CGH->MStreamStorage),
          std::move(impl->MAuxiliaryResources), impl->MCGType, {},
          impl->MKernelIsCooperative, impl->MKernelUsesClusterLaunch,
          impl->MKernelWorkGroupMemorySize, CGH->MCodeLoc));
      break;
    }
    default:
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Unhandled type of command group");
    }

    return CommandGroup;
  }
};

const sycl::detail::KernelArgMask *getKernelArgMaskFromBundle(
    const sycl::kernel_bundle<sycl::bundle_state::input> &KernelBundle,
    std::shared_ptr<sycl::detail::queue_impl> QueueImpl) {

  auto ExecBundle = sycl::link(sycl::compile(KernelBundle));
  EXPECT_FALSE(ExecBundle.empty()) << "Expect non-empty exec kernel bundle";

  // Emulating processing of command group function
  MockHandler MockCGH(QueueImpl);
  MockCGH.use_kernel_bundle(ExecBundle);
  MockCGH.single_task<EAMTestKernel>([] {}); // Actual kernel does not matter

  std::unique_ptr<sycl::detail::CG> CmdGroup = MockCGH.finalize();
  auto *ExecKernel = static_cast<sycl::detail::CGExecKernel *>(CmdGroup.get());

  const auto &KernelBundleImplPtr = ExecKernel->getKernelBundle();
  EXPECT_TRUE(KernelBundleImplPtr)
      << "Expect command group to contain kernel bundle";

  auto KernelID = sycl::detail::ProgramManager::getInstance().getSYCLKernelID(
      ExecKernel->MKernelName);
  sycl::kernel SyclKernel =
      KernelBundleImplPtr->get_kernel(KernelID, KernelBundleImplPtr);
  auto SyclKernelImpl = sycl::detail::getSyclObjImpl(SyclKernel);
  std::shared_ptr<sycl::detail::device_image_impl> DeviceImageImpl =
      SyclKernelImpl->getDeviceImage();
  ur_program_handle_t Program = DeviceImageImpl->get_ur_program_ref();

  EXPECT_TRUE(nullptr == ExecKernel->MSyclKernel ||
              !ExecKernel->MSyclKernel->isCreatedFromSource());

  return sycl::detail::ProgramManager::getInstance().getEliminatedKernelArgMask(
      Program, ExecKernel->MKernelName);
}

// After both kernels are compiled ProgramManager.NativePrograms contains info
// about each UR program handle. However, the result of the linkage of these
// kernels isn't stored in ProgramManager.NativePrograms.
// Check that eliminated arg mask can be found for one of kernels in a
// kernel bundle after two kernels are compiled and linked.
TEST(EliminatedArgMask, KernelBundleWith2Kernels) {
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt = sycl::platform();
  mock::getCallbacks().set_before_callback("urProgramCreateWithIL",
                                           &redefinedProgramCreateEAM);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::queue Queue{Dev};

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          Queue.get_context(), {Dev},
          {sycl::get_kernel_id<EAMTestKernel>(),
           sycl::get_kernel_id<EAMTestKernel2>()});

  const sycl::detail::KernelArgMask *EliminatedArgMask =
      getKernelArgMaskFromBundle(KernelBundle,
                                 sycl::detail::getSyclObjImpl(Queue));
  assert(EliminatedArgMask && "EliminatedArgMask must be not null");

  sycl::detail::KernelArgMask ExpElimArgMask(EAMTestKernelNumArgs);
  ExpElimArgMask[0] = ExpElimArgMask[2] = true;

  EXPECT_EQ(*EliminatedArgMask, ExpElimArgMask);
}

std::vector<std::unique_ptr<mock::dummy_handle_t_>> UsedProgramHandles;
std::vector<std::unique_ptr<mock::dummy_handle_t_>> ProgramHandlesToReuse;
inline ur_result_t setFixedProgramPtr(void *pParams) {
  auto params = *static_cast<ur_program_create_with_il_params_t *>(pParams);
  if (ProgramHandlesToReuse.size())
  {
    auto it = ProgramHandlesToReuse.begin()+1;
    std::move(ProgramHandlesToReuse.begin(), it, std::back_inserter(UsedProgramHandles));
    ProgramHandlesToReuse.erase(ProgramHandlesToReuse.begin(), it);
  }
  else
    UsedProgramHandles.push_back(
        std::make_unique<mock::dummy_handle_t_>(sizeof(unsigned)));
  **params.pphProgram =
      reinterpret_cast<ur_program_handle_t>(UsedProgramHandles.back().get());
  return UR_RESULT_SUCCESS;
}
inline ur_result_t releaseFixedProgramPtr(void *pParams) {
  auto params = *static_cast<ur_program_release_params_t *>(pParams);
  {
    auto it = std::find_if(
        UsedProgramHandles.begin(), UsedProgramHandles.end(),
        [&params](const std::unique_ptr<mock::dummy_handle_t_> &item) {
          return reinterpret_cast<ur_program_handle_t>(item.get()) ==
                 *params.phProgram;
        });
    if (it == UsedProgramHandles.end())
      return UR_RESULT_SUCCESS;
    std::move(it, it + 1, std::back_inserter(ProgramHandlesToReuse));
    UsedProgramHandles.erase(it, it +1);
  }
  return UR_RESULT_SUCCESS;
}

inline ur_result_t customProgramRetain(void *pParams) {
 // do nothing
  return UR_RESULT_SUCCESS;
}

class ProgramManagerTest {
public:
  static std::unordered_multimap<ur_program_handle_t,
                                 const sycl::detail::RTDeviceBinaryImage *> &
  getNativePrograms() {
    return sycl::detail::ProgramManager::getInstance().NativePrograms;
  }
};

// It's possible for the same handle to be reused for multiple distinct programs
// This can happen if a program is released (freeing underlying memory) and then
// a new program happens to get given that same memory for its handle.
// The ProgramContext stores a map with `ur_program_handle_t`s, which are never
// cleared. This test ensures that newer `ur_program_handle_t`s with the same
// values override older ones.
TEST(EliminatedArgMask, ReuseOfHandleValues) {
  sycl::detail::ProgramManager &PM =
      sycl::detail::ProgramManager::getInstance();
  auto &NativePrograms = ProgramManagerTest::getNativePrograms();

  ur_program_handle_t ProgBefore = nullptr;
  ur_program_handle_t ProgAfter = nullptr;
  {
    auto Name = sycl::detail::KernelInfo<EAMTestKernel>::getName();
    sycl::unittest::UrMock<> Mock;
    sycl::platform Plt = sycl::platform();
    mock::getCallbacks().set_replace_callback("urProgramCreateWithIL",
                                              &setFixedProgramPtr);
    mock::getCallbacks().set_replace_callback("urProgramRelease",
                                              &releaseFixedProgramPtr);
    mock::getCallbacks().set_replace_callback("urProgramRetain",
                                              &customProgramRetain);

    const sycl::device Dev = Plt.get_devices()[0];
    sycl::queue Queue{Dev};
    auto Ctx = Queue.get_context();
    ProgBefore = PM.getBuiltURProgram(sycl::detail::getSyclObjImpl(Ctx),
                                      sycl::detail::getSyclObjImpl(Dev), Name);
    auto Mask = PM.getEliminatedKernelArgMask(ProgBefore, Name);
    EXPECT_NE(Mask, nullptr);
    EXPECT_EQ(Mask->at(0), 1);
    EXPECT_EQ(UsedProgramHandles.size(), 1u);
    EXPECT_EQ(NativePrograms.count(ProgBefore), 1u);
  }

  EXPECT_EQ(UsedProgramHandles.size(), 0u);

  {
    auto Name = sycl::detail::KernelInfo<EAMTestKernel3>::getName();
    sycl::unittest::UrMock<> Mock;
    sycl::platform Plt = sycl::platform();
    mock::getCallbacks().set_replace_callback("urProgramCreateWithIL",
                                              &setFixedProgramPtr);
    mock::getCallbacks().set_replace_callback("urProgramRelease",
                                              &releaseFixedProgramPtr);
    mock::getCallbacks().set_replace_callback("urProgramRetain",
                                            &customProgramRetain);

    const sycl::device Dev = Plt.get_devices()[0];
    sycl::queue Queue{Dev};
    auto Ctx = Queue.get_context();
    ProgAfter = PM.getBuiltURProgram(sycl::detail::getSyclObjImpl(Ctx),
                                     sycl::detail::getSyclObjImpl(Dev), Name);
    auto Mask = PM.getEliminatedKernelArgMask(ProgAfter, Name);
    EXPECT_NE(Mask, nullptr);
    EXPECT_EQ(Mask->at(0), 0);
    EXPECT_EQ(UsedProgramHandles.size(), 1u);
    EXPECT_EQ(NativePrograms.count(ProgBefore), 1u);
  }

  // Verify that the test is behaving correctly and that the pointer is being
  // reused
  EXPECT_EQ(ProgBefore, ProgAfter);
}
