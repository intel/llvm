// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// FIXME: ACC devices use emulation path, which is not yet supported

// HIP backend does not currently implement linking.
// XFAIL: hip

// This test checks that specialization constant information is available on
// kernel bundles produced by sycl::link.

#include <sycl/sycl.hpp>

#include <optional>
#include <vector>

using namespace sycl;

class Kernel1;
class Kernel2;

struct TestStruct {
  float f;
  long long ll;
};

bool operator==(const TestStruct &LHS, const TestStruct &RHS) {
  return LHS.f == RHS.f && LHS.ll == RHS.ll;
}
bool operator!=(const TestStruct &LHS, const TestStruct &RHS) {
  return !(LHS == RHS);
}

constexpr specialization_id<TestStruct> SpecName1;
constexpr specialization_id<int> SpecName2;

std::optional<device> FindDeviceWithOnlineLinker() {
  std::vector<device> Devices = device::get_devices();
  auto Device =
      std::find_if(Devices.begin(), Devices.end(), [](const device &D) {
        return D.has(aspect::online_linker);
      });
  if (Device == Devices.end())
    return std::nullopt;
  return *Device;
}

int main() {
  std::optional<device> Device = FindDeviceWithOnlineLinker();

  if (!Device) {
    std::cout << "No device with aspect::online_linker. Skipping test."
              << std::endl;
    return 0;
  }

  context Ctx{*Device};

  kernel_bundle<bundle_state::input> Bundle1 =
      sycl::get_kernel_bundle<Kernel1, bundle_state::input>(Ctx, {*Device});
  kernel_bundle<bundle_state::input> Bundle2 =
      sycl::get_kernel_bundle<Kernel2, bundle_state::input>(Ctx, {*Device});

  assert(Bundle1.has_specialization_constant<SpecName1>());
  assert(Bundle2.has_specialization_constant<SpecName2>());
  assert(!Bundle1.has_specialization_constant<SpecName2>());
  assert(!Bundle2.has_specialization_constant<SpecName1>());

  TestStruct Value1{3.14f, 1234ll};
  int Value2 = 42;
  Bundle1.set_specialization_constant<SpecName1>(Value1);
  Bundle2.set_specialization_constant<SpecName2>(Value2);

  kernel_bundle<bundle_state::object> ObjBundle1 = sycl::compile(Bundle1);
  kernel_bundle<bundle_state::object> ObjBundle2 = sycl::compile(Bundle2);

  assert(ObjBundle1.has_specialization_constant<SpecName1>());
  assert(ObjBundle2.has_specialization_constant<SpecName2>());
  assert(!ObjBundle1.has_specialization_constant<SpecName2>());
  assert(!ObjBundle2.has_specialization_constant<SpecName1>());

  kernel_bundle<bundle_state::executable> LinkedBundle =
      sycl::link({ObjBundle1, ObjBundle2});

  assert(LinkedBundle.has_specialization_constant<SpecName1>());
  assert(LinkedBundle.has_specialization_constant<SpecName2>());

  int Failures = 0;

  if (LinkedBundle.get_specialization_constant<SpecName1>() != Value1) {
    std::cout
        << "Read value of SpecName1 on host is not the same as was written."
        << std::endl;
    ++Failures;
  }

  if (LinkedBundle.get_specialization_constant<SpecName2>() != Value2) {
    std::cout
        << "Read value of SpecName2 on host is not the same as was written."
        << std::endl;
    ++Failures;
  }

  TestStruct ReadValue1;
  int ReadValue2;

  {
    queue Q{Ctx, *Device};

    buffer<TestStruct> ReadValue1Buffer{&ReadValue1, 1};
    Q.submit([&](handler &CGH) {
      CGH.use_kernel_bundle(LinkedBundle);
      accessor ReadValue1Acc{ReadValue1Buffer, CGH, sycl::write_only};
      CGH.single_task<Kernel1>([=](kernel_handler KHR) {
        ReadValue1Acc[0] = KHR.get_specialization_constant<SpecName1>();
      });
    });

    buffer<int> ReadValue2Buffer{&ReadValue2, 1};
    Q.submit([&](handler &CGH) {
      CGH.use_kernel_bundle(LinkedBundle);
      accessor ReadValue2Acc{ReadValue2Buffer, CGH, sycl::write_only};
      CGH.single_task<Kernel2>([=](kernel_handler KHR) {
        ReadValue2Acc[0] = KHR.get_specialization_constant<SpecName2>();
      });
    });
  }

  if (ReadValue1 != Value1) {
    std::cout
        << "Read value of SpecName1 on device is not the same as was written."
        << std::endl;
    ++Failures;
  }

  if (ReadValue2 != Value2) {
    std::cout
        << "Read value of SpecName2 on device is not the same as was written."
        << std::endl;
    ++Failures;
  }

  return Failures;
}
