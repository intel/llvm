// RUN: %clangxx -fsycl -fsyntax-only %s

// Tests the type of the accessor_ptr member alias of sycl::accessor and
// sycl::local_accessor.

#include <sycl/sycl.hpp>

#include <type_traits>

template <typename AccessorT, sycl::access::address_space ExpectedSpace>
void CheckAccessor() {
  static_assert(std::is_same_v<
                typename AccessorT::template accessor_ptr<
                    sycl::access::decorated::legacy>,
                sycl::multi_ptr<typename AccessorT::value_type, ExpectedSpace,
                                sycl::access::decorated::legacy>>);
  static_assert(std::is_same_v<
                typename AccessorT::template accessor_ptr<
                    sycl::access::decorated::yes>,
                sycl::multi_ptr<typename AccessorT::value_type, ExpectedSpace,
                                sycl::access::decorated::yes>>);
  static_assert(std::is_same_v<
                typename AccessorT::template accessor_ptr<
                    sycl::access::decorated::no>,
                sycl::multi_ptr<typename AccessorT::value_type, ExpectedSpace,
                                sycl::access::decorated::no>>);
}

template <typename DataT, int Dims, sycl::access::mode Mode>
void CheckDeviceAccessor() {
  using DeviceAccessorT =
      sycl::accessor<DataT, Dims, Mode, sycl::access::target::device>;
  CheckAccessor<DeviceAccessorT, sycl::access::address_space::global_space>();
}

template <typename DataT, int Dims> void CheckLocalAccessor() {
  using DeviceAccessorT = sycl::local_accessor<DataT, Dims>;
  CheckAccessor<DeviceAccessorT, sycl::access::address_space::local_space>();
}

template <typename DataT, int Dims> void CheckAccessorForModes() {
  CheckDeviceAccessor<DataT, Dims, sycl::access::mode::read>();
  CheckDeviceAccessor<DataT, Dims, sycl::access::mode::read_write>();
  CheckDeviceAccessor<DataT, Dims, sycl::access::mode::write>();
  CheckLocalAccessor<DataT, Dims>();
}

template <typename DataT> void CheckAccessorForAllDimsAndModes() {
  CheckAccessorForModes<DataT, 1>();
  CheckAccessorForModes<DataT, 2>();
  CheckAccessorForModes<DataT, 3>();
}

int main() {
  CheckAccessorForAllDimsAndModes<int>();
  CheckAccessorForAllDimsAndModes<const int>();
  return 0;
}
