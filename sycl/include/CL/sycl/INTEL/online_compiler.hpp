//===------- online_compiler.hpp - Online source compilation service ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/defines_elementary.hpp> // for __SYCL_INLINE_NAMESPACE
#include <CL/sycl/device.hpp>

#include <memory>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace INTEL {

using byte = unsigned char;

enum class compiled_code_format {
  spir_v // the only format supported for now
};

class device_arch {
public:
  static constexpr int any = 0;

  device_arch(int Val) : Val(Val) {}

  enum gpu {
    gpu_any = 1,
    gpu_gen9,
    gpu_skl = gpu_gen9,
    gpu_gen9_5,
    gpu_kbl = gpu_gen9_5,
    gpu_cfl = gpu_gen9_5,
    gpu_gen11,
    gpu_icl = gpu_gen11,
    gpu_gen12
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
  // TBD
};

/// Designates a source language for the online compiler.
enum class source_language { opencl_c, cm };

/// Represents an online compiler for the language given as template
/// parameter.
template <source_language Lang> class online_compiler {
public:
  /// Constructs online compiler which can target any device and produces
  /// given compiled code format. Produces device code is 64-bit.
  /// The created compiler is "optimistic" - it assumes all applicable SYCL
  /// device capabilities are supported by the target device(s).
  online_compiler(compiled_code_format fmt = compiled_code_format::spir_v)
      : OutputFormat(fmt), OutputFormatVersion({0, 0}),
        DeviceArch(device_arch::any), Is64Bit(true), DeviceStepping("") {}

  /// Constructs online compiler which targets given architecture and produces
  /// given compiled code format. Produces device code is 64-bit.
  /// Throws online_compile_error if values of constructor arguments are
  /// contradictory or not supported - e.g. if the source language is not
  /// supported for given device type.
  online_compiler(sycl::info::device_type dev_type, device_arch arch,
                  compiled_code_format fmt = compiled_code_format::spir_v)
      : OutputFormat(fmt), OutputFormatVersion({0, 0}), DeviceArch(arch),
        Is64Bit(true), DeviceStepping("") {}

  /// Constructs online compiler for the target specified by given SYCL device.
  online_compiler(const sycl::device &dev);

  /// Compiles given in-memory \c Lang source to a binary blob. Blob format,
  /// other parameters are set in the constructor by the compilation target
  /// specification parameters.
  /// Specialization for each language will provide exact signatures, which
  /// can be different for different languages.
  /// Throws online_compile_error if compilation is not successful.
  template <typename... Tys>
  std::vector<byte> compile(const std::string &src, const Tys &... args);

  /// Sets the compiled code format of the compilation target and returns *this.
  online_compiler<Lang> &setOutputFormat(compiled_code_format fmt);

  /// Sets the compiled code format version of the compilation target and
  /// returns *this.
  online_compiler<Lang> &setOutputFormatVersion(int major, int minor);

  /// Sets the device type of the compilation target and returns *this.
  online_compiler<Lang> &setTargetDeviceType(sycl::info::device_type type);

  /// Sets the device architecture of the compilation target and returns *this.
  online_compiler<Lang> &setTargetDeviceArch(device_arch arch);

  /// Makes the compilation target 32-bit and returns *this.
  online_compiler<Lang> &set32bitTarget();

  /// Makes the compilation target 64-bit and returns *this.
  online_compiler<Lang> &set64bitTarget();

  /// Sets implementation-defined target device stepping of the compilation
  /// target and returns *this.
  online_compiler<Lang> &setTargetDeviceStepping(const std::string &id);

private:
  // Compilation target specification fields: {

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
  // }
};

// Specializations of the online_compiler class and 'compile' function for
// particular languages and parameter types.

/// Compiles given OpenCL source. May throw \c online_compile_error.
/// @param src - contents of the source
template <>
template <>
std::vector<byte>
online_compiler<source_language::opencl_c>::compile(const std::string &src) {
  // real implementation will call some non-templated impl function here
  return std::vector<byte>{};
}

/// Compiles given OpenCL source. May throw \c online_compile_error.
/// @param src - contents of the source
/// @param options - compilation options (implementation defined); standard
///   OpenCL JIT compiler options must be supported
template <>
template <>
std::vector<byte> online_compiler<source_language::opencl_c>::compile(
    const std::string &src, const std::vector<std::string> &options) {
  // real implementation will call some non-templated impl function here
  return std::vector<byte>{};
}

/// Compiles given CM source.
template <>
template <>
std::vector<byte>
online_compiler<source_language::cm>::compile(const std::string &src) {
  // real implementation will call some non-templated impl function here
  return std::vector<byte>{};
}

/// Compiles given CM source.
/// @param options - compilation options (implementation defined)
template <>
template <>
std::vector<byte> online_compiler<source_language::cm>::compile(
    const std::string &src, const std::vector<std::string> &options) {
  // real implementation will call some non-templated impl function here
  return std::vector<byte>{};
}

} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
