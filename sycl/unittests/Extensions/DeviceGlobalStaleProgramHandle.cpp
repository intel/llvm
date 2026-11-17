// A device_global initializer registered for the pre-link program of a JIT
// compile+link must not survive its release: a later unrelated program that
// gets the same handle value must not have that device_global written into it.

#include <sycl/sycl.hpp>

#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

class StaleInitLinkedKernel;
class StaleInitDepKernel;
class StaleInitPlainKernel;
constexpr const char *StaleInitLinkedKernelName = "StaleInitLinkedKernel";
constexpr const char *StaleInitDepKernelName = "StaleInitDepKernel";
constexpr const char *StaleInitPlainKernelName = "StaleInitPlainKernel";
constexpr const char *StaleInitDeviceGlobalName = "StaleInitDeviceGlobal";
constexpr const char *StaleInitDepSymbol = "StaleInitDepSymbol";

sycl::ext::oneapi::experimental::device_global<int> StaleInitDeviceGlobal;

namespace sycl {
inline namespace _V1 {
namespace detail {
template <>
struct KernelInfo<StaleInitLinkedKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return StaleInitLinkedKernelName; }
};
template <>
struct KernelInfo<StaleInitDepKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return StaleInitDepKernelName; }
};
template <>
struct KernelInfo<StaleInitPlainKernel> : public unittest::MockKernelInfoBase {
  static constexpr const char *getName() { return StaleInitPlainKernelName; }
};
} // namespace detail
} // namespace _V1
} // namespace sycl

namespace {

std::vector<sycl::unittest::MockProperty>
makeSymbolProps(const std::vector<std::string> &Symbols) {
  std::vector<sycl::unittest::MockProperty> Props;
  for (const std::string &Symbol : Symbols) {
    std::vector<char> Storage(sizeof(uint32_t));
    uint32_t Val = 1;
    auto *DataPtr = reinterpret_cast<char *>(&Val);
    std::uninitialized_copy(DataPtr, DataPtr + sizeof(uint32_t),
                            Storage.data());
    Props.emplace_back(Symbol, Storage, SYCL_PROPERTY_TYPE_UINT32);
  }
  return Props;
}

// Image with a device_global that imports a symbol, so it is JIT-built via
// compile + link and its pre-link program is released afterwards.
sycl::unittest::MockDeviceImage generateLinkedImage() {
  using namespace sycl::unittest;
  sycl::detail::device_global_map::add(&StaleInitDeviceGlobal,
                                       StaleInitDeviceGlobalName);
  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_DEVICE_GLOBALS,
                 std::vector<MockProperty>{makeDeviceGlobalInfo(
                     StaleInitDeviceGlobalName, sizeof(int), 0)});
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_IMPORTED_SYMBOLS,
                 makeSymbolProps({StaleInitDepSymbol}));
  return MockDeviceImage(makeEmptyKernels({StaleInitLinkedKernelName}),
                         std::move(PropSet));
}

sycl::unittest::MockDeviceImage generateDepImage() {
  using namespace sycl::unittest;
  MockPropertySet PropSet;
  PropSet.insert(__SYCL_PROPERTY_SET_SYCL_EXPORTED_SYMBOLS,
                 makeSymbolProps({StaleInitDepSymbol}));
  return MockDeviceImage(makeEmptyKernels({StaleInitDepKernelName}),
                         std::move(PropSet));
}

// Image without device_globals; stands in for a program the runtime knows
// nothing about (interop program, oneDNN program, ...).
sycl::unittest::MockDeviceImage generatePlainImage() {
  return sycl::unittest::MockDeviceImage(
      sycl::unittest::makeEmptyKernels({StaleInitPlainKernelName}));
}

sycl::unittest::MockDeviceImage Imgs[] = {
    generateLinkedImage(), generateDepImage(), generatePlainImage()};
sycl::unittest::MockDeviceImageArray<3> ImgArray{Imgs};

int MockDeviceGlobalUSM[1] = {0};
std::vector<ur_program_handle_t> CreatedPrograms;
std::vector<ur_program_handle_t> ReleasedPrograms;
std::vector<std::pair<ur_program_handle_t, std::string>> DeviceGlobalWrites;
// When set, the next urProgramCreateWithIL returns this handle value, which
// models the adapter reusing the storage of a released program.
ur_program_handle_t RecycledHandle = nullptr;

ur_result_t replace_urProgramCreateWithIL(void *pParams) {
  auto params = *static_cast<ur_program_create_with_il_params_t *>(pParams);
  ur_program_handle_t Handle =
      RecycledHandle ? RecycledHandle
                     : mock::createDummyHandle<ur_program_handle_t>();
  RecycledHandle = nullptr;
  **params.pphProgram = Handle;
  CreatedPrograms.push_back(Handle);
  return UR_RESULT_SUCCESS;
}

// Record releases but keep the dummy handle alive so it can be recycled.
ur_result_t replace_urProgramRelease(void *pParams) {
  auto params = *static_cast<ur_program_release_params_t *>(pParams);
  ReleasedPrograms.push_back(*params.phProgram);
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urEnqueueDeviceGlobalVariableWrite(void *pParams) {
  auto params =
      *static_cast<ur_enqueue_device_global_variable_write_params_t *>(pParams);
  DeviceGlobalWrites.emplace_back(*params.phProgram, *params.pname);
  return UR_RESULT_SUCCESS;
}

ur_result_t after_urUSMDeviceAlloc(void *pParams) {
  auto params = *static_cast<ur_usm_device_alloc_params_t *>(pParams);
  **params.pppMem = MockDeviceGlobalUSM;
  return UR_RESULT_SUCCESS;
}

bool wasReleased(ur_program_handle_t Handle) {
  return std::find(ReleasedPrograms.begin(), ReleasedPrograms.end(), Handle) !=
         ReleasedPrograms.end();
}

} // namespace

TEST(DeviceGlobalStaleProgramHandle, PreLinkInitializerNotInherited) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urProgramCreateWithIL",
                                            &replace_urProgramCreateWithIL);
  mock::getCallbacks().set_replace_callback("urProgramRelease",
                                            &replace_urProgramRelease);
  mock::getCallbacks().set_after_callback(
      "urEnqueueDeviceGlobalVariableWrite",
      &after_urEnqueueDeviceGlobalVariableWrite);
  mock::getCallbacks().set_after_callback("urUSMDeviceAlloc",
                                          &after_urUSMDeviceAlloc);
  CreatedPrograms.clear();
  ReleasedPrograms.clear();
  DeviceGlobalWrites.clear();
  RecycledHandle = nullptr;

  sycl::platform Plt = sycl::platform();
  sycl::queue Q(Plt.get_devices()[0]);

  // Main image + dependency are created, linked into a third program, and the
  // two inputs are released. The device_global is written into the linked
  // program only.
  Q.single_task<StaleInitLinkedKernel>([]() {}).wait();
  ASSERT_EQ(CreatedPrograms.size(), 2u);
  ur_program_handle_t PreLinkProgram = CreatedPrograms[0];
  ASSERT_TRUE(wasReleased(PreLinkProgram))
      << "pre-link program expected to be released after linking";
  ASSERT_EQ(DeviceGlobalWrites.size(), 1u);
  EXPECT_NE(DeviceGlobalWrites[0].first, PreLinkProgram);
  EXPECT_EQ(DeviceGlobalWrites[0].second, StaleInitDeviceGlobalName);

  // A program without device_globals comes back with the released handle
  // value. It must not receive the pre-link program's initializer.
  RecycledHandle = PreLinkProgram;
  Q.single_task<StaleInitPlainKernel>([]() {}).wait();
  ASSERT_EQ(CreatedPrograms.size(), 3u);
  ASSERT_EQ(CreatedPrograms[2], PreLinkProgram);
  EXPECT_EQ(DeviceGlobalWrites.size(), 1u)
      << "device_global of a released pre-link program was written into an "
         "unrelated program that reused its handle value";
}
