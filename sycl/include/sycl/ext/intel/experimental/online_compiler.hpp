//===------- online_compiler.hpp - Online source compilation service ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/export.hpp>             // for __SYCL_EXPORT
#include <sycl/device.hpp>

#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::experimental {

using byte = unsigned char;

enum class compiled_code_format {
  spir_v = 0 // the only format supported for now
};

class device_arch {
public:
  static constexpr int any = 0;

  device_arch(int Val) : Val(Val) {}

  // TODO1: the list must be extended with a bunch of new GPUs available.
  // TODO2: the list of supported GPUs grows rapidly.
  // The API must allow user to define the target GPU option even if it is
  // not listed in this enumerator below.
  enum gpu {
    gpu_any = 1,
    gpu_gen9 = 2,
    gpu_skl = gpu_gen9,
    gpu_gen9_5 = 3,
    gpu_kbl = gpu_gen9_5,
    gpu_cfl = gpu_gen9_5,
    gpu_gen11 = 4,
    gpu_icl = gpu_gen11,
    gpu_gen12 = 5,
    gpu_tgl = gpu_gen12,
    gpu_tgllp = gpu_gen12
  };

  enum cpu {
    cpu_any = 1,
  };

  enum fpga {
    fpga_any = 1,
  };

  operator int() { return Val; }

private:
  int Val;
};

/// Represents an error happend during online compilation.
class online_compile_error : public sycl::exception {
public:
  online_compile_error() = default;
  online_compile_error(const std::string &Msg)
      : sycl::exception(make_error_code(errc::invalid), Msg) {}
};

/// Designates a source language for the online compiler.
enum class source_language { opencl_c = 0, cm = 1 };

/// Represents an online compiler for the language given as template
/// parameter.
template <source_language Lang>
class __SYCL2020_DEPRECATED(
    "experimental online_compiler is being deprecated. See "
    "'sycl_ext_oneapi_kernel_compiler.asciidoc' instead for new kernel "
    "compiler extension to kernel_bundle implementation.") online_compiler {
public:
  /// Constructs online compiler which can target any device and produces
  /// given compiled code format. Produces 64-bit device code.
  /// The created compiler is "optimistic" - it assumes all applicable SYCL
  /// device capabilities are supported by the target device(s).
  online_compiler(compiled_code_format fmt = compiled_code_format::spir_v)
      : OutputFormat(fmt), OutputFormatVersion({0, 0}),
        DeviceType(sycl::info::device_type::all), DeviceArch(device_arch::any),
        Is64Bit(true), DeviceStepping("") {}

  /// Constructs online compiler which targets given architecture and produces
  /// given compiled code format. Produces 64-bit device code.
  /// Throws online_compile_error if values of constructor arguments are
  /// contradictory or not supported - e.g. if the source language is not
  /// supported for given device type.
  online_compiler(sycl::info::device_type dev_type, device_arch arch,
                  compiled_code_format fmt = compiled_code_format::spir_v)
      : OutputFormat(fmt), OutputFormatVersion({0, 0}), DeviceType(dev_type),
        DeviceArch(arch), Is64Bit(true), DeviceStepping("") {}

  /// Constructs online compiler for the target specified by given SYCL device.
  // TODO: the initial version generates the generic code (SKL now), need
  // to do additional device::info calls to determine the device by it's
  // features.
  online_compiler(const sycl::device &)
      : OutputFormat(compiled_code_format::spir_v), OutputFormatVersion({0, 0}),
        DeviceType(sycl::info::device_type::all), DeviceArch(device_arch::any),
        Is64Bit(true), DeviceStepping("") {}

  /// Compiles given in-memory \c Lang source to a binary blob. Blob format,
  /// other parameters are set in the constructor by the compilation target
  /// specification parameters.
  /// Specialization for each language will provide exact signatures, which
  /// can be different for different languages.
  /// Throws online_compile_error if compilation is not successful.
  template <typename... Tys>
  std::vector<byte> compile(const std::string &src, const Tys &...args);

  /// Sets the compiled code format of the compilation target and returns *this.
  online_compiler<Lang> &setOutputFormat(compiled_code_format fmt) {
    OutputFormat = fmt;
    return *this;
  }

  /// Sets the compiled code format version of the compilation target and
  /// returns *this.
  online_compiler<Lang> &setOutputFormatVersion(int major, int minor) {
    OutputFormatVersion = {major, minor};
    return *this;
  }

  /// Sets the device type of the compilation target and returns *this.
  online_compiler<Lang> &setTargetDeviceType(sycl::info::device_type type) {
    DeviceType = type;
    return *this;
  }

  /// Sets the device architecture of the compilation target and returns *this.
  online_compiler<Lang> &setTargetDeviceArch(device_arch arch) {
    DeviceArch = arch;
    return *this;
  }

  /// Makes the compilation target 32-bit and returns *this.
  online_compiler<Lang> &set32bitTarget() {
    Is64Bit = false;
    return *this;
  };

  /// Makes the compilation target 64-bit and returns *this.
  online_compiler<Lang> &set64bitTarget() {
    Is64Bit = true;
    return *this;
  };

  /// Sets implementation-defined target device stepping of the compilation
  /// target and returns *this.
  online_compiler<Lang> &setTargetDeviceStepping(const std::string &id) {
    DeviceStepping = id;
    return *this;
  }

private:
  /// Compiled code format.
  compiled_code_format OutputFormat;

  /// Compiled code format version - a pair of "major" and "minor" components
  std::pair<int, int> OutputFormatVersion;

  /// Target device type
  sycl::info::device_type DeviceType;

  /// Target device architecture
  device_arch DeviceArch;

  /// Whether the target device architecture is 64-bit
  bool Is64Bit;

  /// Target device stepping (implementation defined)
  std::string DeviceStepping;

  /// Handles to helper functions used by the implementation.
  void *CompileToSPIRVHandle = nullptr;
  void *FreeSPIRVOutputsHandle = nullptr;
};

// Specializations of the online_compiler class and 'compile' function for
// particular languages and parameter types.

/// Compiles the given OpenCL source. May throw \c online_compile_error.
/// @param src - contents of the source.
/// @param options - compilation options (implementation defined); standard
///   OpenCL JIT compiler options must be supported.
template <>
template <>
__SYCL_EXPORT std::vector<byte>
online_compiler<source_language::opencl_c>::compile(
    const std::string &src, const std::vector<std::string> &options);

/// Compiles the given OpenCL source. May throw \c online_compile_error.
/// @param src - contents of the source.
template <>
template <>
std::vector<byte>
online_compiler<source_language::opencl_c>::compile(const std::string &src) {
  return compile(src, std::vector<std::string>{});
}

/// Compiles the given CM source \p src.
/// @param src - contents of the source.
/// @param options - compilation options (implementation defined).
template <>
template <>
__SYCL_EXPORT std::vector<byte> online_compiler<source_language::cm>::compile(
    const std::string &src, const std::vector<std::string> &options);

/// Compiles the given CM source \p src.
template <>
template <>
std::vector<byte>
online_compiler<source_language::cm>::compile(const std::string &src) {
  return compile(src, std::vector<std::string>{});
}

} // namespace ext::intel::experimental
} // namespace _V1
} // namespace sycl
