//==----- device_binary_image.hpp --- SYCL device binary image abstraction -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>

#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// SYCL RT wrapper over PI binary image.
class RTDeviceBinaryImage : public pi::DeviceBinaryImage {
public:
  RTDeviceBinaryImage(OSModuleHandle ModuleHandle)
      : pi::DeviceBinaryImage(), ModuleHandle(ModuleHandle) {}
  RTDeviceBinaryImage(pi_device_binary Bin, OSModuleHandle ModuleHandle)
      : pi::DeviceBinaryImage(Bin), ModuleHandle(ModuleHandle) {}
  // Explicitly delete copy constructor/operator= to avoid unintentional copies
  RTDeviceBinaryImage(const RTDeviceBinaryImage &) = delete;
  RTDeviceBinaryImage &operator=(const RTDeviceBinaryImage &) = delete;
  // Explicitly retain move constructors to facilitate potential moves across
  // collections
  RTDeviceBinaryImage(RTDeviceBinaryImage &&) = default;
  RTDeviceBinaryImage &operator=(RTDeviceBinaryImage &&) = default;

  OSModuleHandle getOSModuleHandle() const { return ModuleHandle; }

  ~RTDeviceBinaryImage() override {}

  bool supportsSpecConstants() const {
    return getFormat() == PI_DEVICE_BINARY_TYPE_SPIRV;
  }

  const pi_device_binary_struct &getRawData() const { return *get(); }

  void print() const override {
    pi::DeviceBinaryImage::print();
    std::cerr << "    OSModuleHandle=" << ModuleHandle << "\n";
  }

protected:
  OSModuleHandle ModuleHandle;
};

// Dynamically allocated device binary image, which de-allocates its binary
// data in destructor.
class DynRTDeviceBinaryImage : public RTDeviceBinaryImage {
public:
  DynRTDeviceBinaryImage(std::unique_ptr<char[]> &&DataPtr, size_t DataSize,
                         OSModuleHandle M);
  ~DynRTDeviceBinaryImage() override;

  void print() const override {
    RTDeviceBinaryImage::print();
    std::cerr << "    DYNAMICALLY CREATED\n";
  }

protected:
  std::unique_ptr<char[]> Data;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
