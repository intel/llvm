//==------- EliminatedArgMask.cpp --- eliminated args mask unit test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/handler_impl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>
#include <sycl/sycl.hpp>

#include <helpers/MockKernelInfo.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class EAMTestKernel;
class EAMTestKernel2;
constexpr const char EAMTestKernelName[] = "EAMTestKernel";
constexpr const char EAMTestKernel2Name[] = "EAMTestKernel2";
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

} // namespace detail
} // namespace _V1
} // namespace sycl

static sycl::unittest::PiImage generateEAMTestKernelImage() {
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
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage generateEAMTestKernel2Image() {
  using namespace sycl::unittest;

  PiPropertySet PropSet;

  std::vector<unsigned char> Bin{6, 7, 8, 9, 10, 11}; // Random data

  PiArray<PiOffloadEntry> Entries = makeEmptyKernels({EAMTestKernel2Name});

  PiImage Img{PI_DEVICE_BINARY_TYPE_SPIRV,            // Format
              __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64, // DeviceTargetSpec
              "",                                     // Compile options
              "",                                     // Link options
              std::move(Bin),
              std::move(Entries),
              std::move(PropSet)};

  return Img;
}

static sycl::unittest::PiImage EAMImg = generateEAMTestKernelImage();
static sycl::unittest::PiImage EAM2Img = generateEAMTestKernel2Image();
static sycl::unittest::PiImageArray<1> EAMImgArray{&EAMImg};
static sycl::unittest::PiImageArray<1> EAM2ImgArray{&EAM2Img};

// pi_program address is used as a key for ProgramManager::NativePrograms
// storage. redefinedProgramLinkCommon makes pi_program address equal to 0x1.
// Make sure that size of Bin is different for device images used in these tests
// and greater than 1.
inline pi_result redefinedProgramCreateEAM(pi_context, const void *, size_t,
                                           pi_program *ret_program) {
  static size_t PiProgramAddr = 2;
  *ret_program = reinterpret_cast<pi_program>(PiProgramAddr++);
  return PI_SUCCESS;
}

class MockHandler : public sycl::handler {

public:
  MockHandler(std::shared_ptr<sycl::detail::queue_impl> Queue)
      : sycl::handler(Queue, /* IsHost */ false, /*CallerNeedsEvent*/ true) {}

  std::unique_ptr<sycl::detail::CG> finalize() {
    auto CGH = static_cast<sycl::handler *>(this);
    std::unique_ptr<sycl::detail::CG> CommandGroup;
    switch (getType()) {
    case sycl::detail::CG::Kernel: {
      CommandGroup.reset(new sycl::detail::CGExecKernel(
          std::move(CGH->MNDRDesc), std::move(CGH->MHostKernel),
          std::move(CGH->MKernel), std::move(MImpl->MKernelBundle),
          std::move(CGH->CGData), std::move(CGH->MArgs),
          CGH->MKernelName.c_str(), std::move(CGH->MStreamStorage),
          std::move(MImpl->MAuxiliaryResources), CGH->MCGType, {},
          MImpl->MKernelIsCooperative, CGH->MCodeLoc));
      break;
    }
    default:
      throw sycl::runtime_error("Unhandled type of command group",
                                PI_ERROR_INVALID_OPERATION);
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
  sycl::detail::pi::PiProgram Program = DeviceImageImpl->get_program_ref();

  EXPECT_TRUE(nullptr == ExecKernel->MSyclKernel ||
              !ExecKernel->MSyclKernel->isCreatedFromSource());

  return sycl::detail::ProgramManager::getInstance().getEliminatedKernelArgMask(
      Program, ExecKernel->MKernelName);
}

// After both kernels are compiled ProgramManager.NativePrograms contains info
// about each pi_program. However, the result of the linkage of these kernels
// isn't stored in ProgramManager.NativePrograms.
// Check that eliminated arg mask can be found for one of kernels in a
// kernel bundle after two kernels are compiled and linked.
TEST(EliminatedArgMask, KernelBundleWith2Kernels) {
  sycl::unittest::PiMock Mock;
  sycl::platform Plt = Mock.getPlatform();
  Mock.redefineBefore<sycl::detail::PiApiKind::piProgramCreate>(
      redefinedProgramCreateEAM);

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
