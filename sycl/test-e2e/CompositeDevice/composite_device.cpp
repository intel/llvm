// RUN: %{build} -o %t.out
// RUN: env ZE_FLAT_DEVICE_HIERARCHY=COMBINED %{run} %t.out
// RUN: env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE %{run} %t.out
// RUN: env ZE_FLAT_DEVICE_HIERARCHY=FLAT %{run} %t.out

#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE

using namespace sycl::ext::oneapi::experimental;

bool isL0Backend(sycl::backend backend) {
  return (backend == sycl::backend::ext_oneapi_level_zero);
}

bool isCombinedMode() {
  char *Mode = nullptr;
  bool Res = false;
#ifdef _WIN32
  size_t Size = 0;
  auto Err = _dupenv_s(&Mode, &Size, "ZE_FLAT_DEVICE_HIERARCHY");
  Res = (Mode != nullptr) && (std::strcmp(Mode, "COMBINED") == 0);
  free(Mode);
#else
  Mode = std::getenv("ZE_FLAT_DEVICE_HIERARCHY");
  Res = (Mode != nullptr) && (std::strcmp(Mode, "COMBINED") == 0);
#endif
  return Res;
}

int main() {
  sycl::queue q;
  bool IsCombined = isCombinedMode();
  auto Platforms = sycl::platform::get_platforms();

  // Check that device::get_devices() and platform::get_devices() do not return
  // composite devices.
  {
    auto Devs = sycl::device::get_devices();
    for (const auto &D : Devs) {
      assert(!D.has(sycl::aspect::ext_oneapi_is_composite));

      // If ZE_FLAT_DEVICE_HIERARCHY != COMBINED,
      // sycl::aspect::ext_oneapi_is_component must be false.
      assert(IsCombined || !D.has(sycl::aspect::ext_oneapi_is_component));
    }

    for (const auto &P : Platforms) {
      bool IsL0 = isL0Backend(P.get_backend());
      if (!IsL0)
        continue;

      Devs = P.get_devices();
      for (const auto &D : Devs) {
        assert(!D.has(sycl::aspect::ext_oneapi_is_composite));

        // If ZE_FLAT_DEVICE_HIERARCHY != COMBINED,
        // sycl::aspect::ext_oneapi_is_component must be false.
        assert(!(!IsCombined && D.has(sycl::aspect::ext_oneapi_is_component)));
      }
    }
  }

  // Check that:
  //   A. The free function get_composite_devices returns all of the composite
  //      devices across all platforms.
  //   B. The member function platform::ext_oneapi_get_composite_devices returns
  //      the composite devices within the given platform.
  //   C. The APIs defined in this extension are only useful when using the
  //      Level Zero backend, and they are only useful when the Level Zero
  //      environment variable ZE_FLAT_DEVICE_HIERARCHY=COMBINED is set. The
  //      APIs may be called even when using other backends, but they will
  //      return an empty list of composite devices.
  //   D. The execution environment for a SYCL application has a fixed number of
  //      composite devices which does not vary as the application executes. As
  //      a result, each call to these functions returns the same set of device
  //      objects, and the order of those objects does not vary between calls.
  {
    std::vector<sycl::device> AllCompositeDevs = get_composite_devices();
    std::vector<sycl::device> CombinedCompositeDevs;
    for (const auto &P : Platforms) {
      auto CompositeDevs = P.ext_oneapi_get_composite_devices();
      bool IsL0 = isL0Backend(P.get_backend());
      // Check C.
      assert(CompositeDevs.empty() || (IsL0 && IsCombined));

      for (const auto &D : CompositeDevs) {
        if (std::find(CombinedCompositeDevs.begin(),
                      CombinedCompositeDevs.end(),
                      D) == CombinedCompositeDevs.end())
          CombinedCompositeDevs.push_back(D);
      }
    }
    // Check A. and B.
    assert(AllCompositeDevs.size() == CombinedCompositeDevs.size());
    assert(std::all_of(AllCompositeDevs.begin(), AllCompositeDevs.end(),
                       [&](const sycl::device &D) {
                         const bool Found =
                             std::find(CombinedCompositeDevs.begin(),
                                       CombinedCompositeDevs.end(),
                                       D) != CombinedCompositeDevs.end();
                         return Found &&
                                D.has(sycl::aspect::ext_oneapi_is_composite);
                       }));

    // Check D.
    std::vector<sycl::device> AllCompositeDevs2 = get_composite_devices();
    std::vector<sycl::device> CombinedCompositeDevs2;
    for (const auto &P : Platforms) {
      auto CompositeDevs = P.ext_oneapi_get_composite_devices();
      bool IsL0 = isL0Backend(P.get_backend());
      // Check C.
      assert(CompositeDevs.empty() || (IsL0 && IsCombined));

      for (const auto &D : CompositeDevs) {
        if (std::find(CombinedCompositeDevs2.begin(),
                      CombinedCompositeDevs2.end(),
                      D) == CombinedCompositeDevs2.end())
          CombinedCompositeDevs2.push_back(D);
      }
    }
    assert(AllCompositeDevs.size() == AllCompositeDevs2.size());
    assert(CombinedCompositeDevs.size() == CombinedCompositeDevs2.size());
    for (size_t i = 0; i < AllCompositeDevs.size(); ++i) {
      assert(AllCompositeDevs[i] == AllCompositeDevs2[i]);
      assert(CombinedCompositeDevs[i] == CombinedCompositeDevs2[i]);
    }
  }

  // Check that device::info::component_devices:
  //   A. Returns the set of component devices that are contained by a composite
  //      device (at least 2).
  //   B. If "this" device is not a composite device, returns an empty vector.
  {
    auto Devs = sycl::device::get_devices();
    for (const auto &D : Devs) {
      //  Check B.
      assert(!D.has(sycl::aspect::ext_oneapi_is_composite));
      auto Components = D.get_info<info::device::component_devices>();
      assert(Components.empty());

      // Check A.
      auto IsComponent = D.has(sycl::aspect::ext_oneapi_is_component);
      // A device can be neither composite nor component. This happens when
      // there are not multiple tiles in a single card.
      if (IsComponent) {
        auto Composite = D.get_info<info::device::composite_device>();
        Components = Composite.get_info<info::device::component_devices>();
        assert(Components.size() >= 2);
      }
    }
  }

  // Check that device::info::composite_device:
  //   A. Returns the composite device which contains this component device.
  //   B. Since the set of composite devices if fixed, returns a device object
  //      which is a copy of one of the device objects returned by
  //      get_composite_devices.
  //   C. Throws a synchronous exception with the errc::invalid error code if
  //      "this" device does not have aspect::ext_oneapi_is_component.
  {
    auto Devs = sycl::device::get_devices();
    for (const auto &D : Devs) {
      bool IsL0 = isL0Backend(D.get_backend());
      if (!IsL0 || !IsCombined)
        continue;
      // Check A.
      assert(!D.has(sycl::aspect::ext_oneapi_is_composite));
      auto IsComponent = D.has(sycl::aspect::ext_oneapi_is_component);
      // A device can be neither composite nor component. This happens when
      // there are not multiple tiles in a single card.
      if (IsComponent) {
        auto Composite = D.get_info<info::device::composite_device>();
        assert(Composite.has(sycl::aspect::ext_oneapi_is_composite));
        // Check B.
        std::vector<sycl::device> AllCompositeDevs = get_composite_devices();
        assert(std::find(AllCompositeDevs.begin(), AllCompositeDevs.end(),
                         Composite) != AllCompositeDevs.end());
        // Check C.
        assert(!Composite.has(sycl::aspect::ext_oneapi_is_component));
        try {
          auto Invalid = Composite.get_info<info::device::composite_device>();
          assert(false && "Exception expected.");
        } catch (sycl::exception &E) {
          assert(E.code() == sycl::errc::invalid &&
                 "errc should be errc::invalid");
        }
      }
    }
  }

  // Check that ext_oneapi_is_component applies only to a root device that is a
  // direct component of some composite device. A sub-device will not have this
  // aspect even if its parent is a component device.
  {
    auto Devs = sycl::device::get_devices();
    for (const auto &D : Devs) {
      bool IsL0 = isL0Backend(D.get_backend());
      if (!IsL0 || !IsCombined)
        continue;

      auto PartitionProperties =
          D.get_info<sycl::info::device::partition_properties>();
      if (PartitionProperties.empty())
        continue;

      std::vector<sycl::device> SubDevices;
      for (const auto &PartitionProperty : PartitionProperties) {
        if (PartitionProperty ==
            sycl::info::partition_property::partition_equally) {
          size_t CompUnits = 2;
          SubDevices = D.create_sub_devices<
              sycl::info::partition_property::partition_equally>(CompUnits);
        } else if (PartitionProperty ==
                   sycl::info::partition_property::partition_by_counts) {
          SubDevices = D.create_sub_devices<
              sycl::info::partition_property::partition_by_counts>(
              std::vector<size_t>{2});
        } else if (PartitionProperty == sycl::info::partition_property::
                                            partition_by_affinity_domain) {
          SubDevices = D.create_sub_devices<
              sycl::info::partition_property::partition_by_affinity_domain>(
              sycl::info::partition_affinity_domain::numa);
        }
      }

      for (const auto &SubDevice : SubDevices) {
        assert(!SubDevice.has(sycl::aspect::ext_oneapi_is_component));
      }
    }
  }
}

#endif // SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
