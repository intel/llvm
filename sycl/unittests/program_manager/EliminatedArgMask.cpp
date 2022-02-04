//==------- EliminatedArgMask.cpp --- eliminated args mask unit test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/commands.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiImage.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

class EAMTestKernel;
class EAMTestKernel2;
const char EAMTestKernelName[] = "EAMTestKernel";
const char EAMTestKernel2Name[] = "EAMTestKernel2";
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

template <> struct KernelInfo<EAMTestKernel2> {
  static constexpr unsigned getNumParams() { return 0; }
  static const kernel_param_desc_t &getParamDesc(int) {
    static kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr const char *getName() { return EAMTestKernel2Name; }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

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
      : sycl::handler(Queue, /* IsHost */ false) {}

  std::unique_ptr<sycl::detail::CG> finalize() {
    auto CGH = static_cast<sycl::handler *>(this);
    std::unique_ptr<sycl::detail::CG> CommandGroup;
    switch (getType()) {
    case sycl::detail::CG::Kernel: {
      CommandGroup.reset(new sycl::detail::CGExecKernel(
          std::move(CGH->MNDRDesc), std::move(CGH->MHostKernel),
          std::move(CGH->MKernel), std::move(CGH->MArgsStorage),
          std::move(CGH->MAccStorage), std::move(CGH->MSharedPtrStorage),
          std::move(CGH->MRequirements), std::move(CGH->MEvents),
          std::move(CGH->MArgs), std::move(CGH->MKernelName),
          std::move(CGH->MOSModuleHandle), std::move(CGH->MStreamStorage),
          CGH->MCGType, CGH->MCodeLoc));
      break;
    }
    default:
      throw sycl::runtime_error("Unhandled type of command group",
                                PI_INVALID_OPERATION);
    }

    return CommandGroup;
  }
};

sycl::detail::ProgramManager::KernelArgMask getKernelArgMaskFromBundle(
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

  auto KernelIDImpl =
      std::make_shared<sycl::detail::kernel_id_impl>(ExecKernel->MKernelName);
  sycl::kernel SyclKernel = KernelBundleImplPtr->get_kernel(
      sycl::detail::createSyclObjFromImpl<sycl::kernel_id>(KernelIDImpl),
      KernelBundleImplPtr);
  auto SyclKernelImpl = sycl::detail::getSyclObjImpl(SyclKernel);
  std::shared_ptr<sycl::detail::device_image_impl> DeviceImageImpl =
      SyclKernelImpl->getDeviceImage();
  sycl::detail::pi::PiProgram Program = DeviceImageImpl->get_program_ref();

  EXPECT_TRUE(nullptr == ExecKernel->MSyclKernel ||
              !ExecKernel->MSyclKernel->isCreatedFromSource());

  return sycl::detail::ProgramManager::getInstance().getEliminatedKernelArgMask(
      ExecKernel->MOSModuleHandle, Program, ExecKernel->MKernelName);
}

// After both kernels are compiled ProgramManager.NativePrograms contains info
// about each pi_program. However, the result of the linkage of these kernels
// isn't stored in ProgramManager.NativePrograms.
// Check that eliminated arg mask can be found for one of kernels in a
// kernel bundle after two kernels are compiled and linked.
TEST(EliminatedArgMask, KernelBundleWith2Kernels) {
  sycl::platform Plt{sycl::default_selector()};
  if (Plt.is_host() || Plt.get_backend() == sycl::backend::ext_oneapi_cuda ||
      Plt.get_backend() == sycl::backend::ext_oneapi_hip) {
    std::cerr << "Test is not supported on "
              << Plt.get_info<sycl::info::platform::name>() << ", skipping\n";
    GTEST_SKIP(); // test is not supported on selected platform.
  }

  sycl::unittest::PiMock Mock{Plt};
  setupDefaultMockAPIs(Mock);
  Mock.redefine<sycl::detail::PiApiKind::piProgramCreate>(
      redefinedProgramCreateEAM);

  const sycl::device Dev = Plt.get_devices()[0];
  sycl::queue Queue{Dev};

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          Queue.get_context(), {Dev},
          {sycl::get_kernel_id<EAMTestKernel>(),
           sycl::get_kernel_id<EAMTestKernel2>()});

  sycl::detail::ProgramManager::KernelArgMask EliminatedArgMask =
      getKernelArgMaskFromBundle(KernelBundle,
                                 sycl::detail::getSyclObjImpl(Queue));

  sycl::detail::ProgramManager::KernelArgMask ExpElimArgMask(
      EAMTestKernelNumArgs);
  ExpElimArgMask[0] = ExpElimArgMask[2] = true;

  EXPECT_EQ(EliminatedArgMask, ExpElimArgMask);
}
