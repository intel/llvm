//==--------------- program.hpp --- SYCL program ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/stl.hpp>

#include <memory>

__SYCL_INLINE namespace cl {
namespace sycl {

// Forward declarations
class context;
class device;
namespace detail {
class program_impl;
}

enum class program_state { none, compiled, linked };

class program {
  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

public:
  program() = delete;

  /// Constructs an instance of SYCL program.
  ///
  /// The program will be created in the program_state::none state and associated
  /// with the provided context and the SYCL devices that are associated with
  /// the context.
  ///
  /// @param Context is an instance of SYCL context.
  explicit program(const context &Context);

  /// Constructs an instance of SYCL program for the provided DeviceList.
  ///
  /// The program will be created in the program_state::none state and associated
  /// with the provided context and the SYCL devices in the provided DeviceList.
  ///
  /// @param Context is an instance of SYCL context.
  /// @param DeviceList is a list of SYCL devices.
  program(const context &context, vector_class<device> DeviceList);

  /// Constructs an instance of SYCL program by linking together each SYCL
  /// program instance in ProgramList.
  ///
  /// Each SYCL program in ProgramList must be in the program_state::compiled
  /// state and must be associated with the same SYCL context. Otherwise an
  /// invalid_object_error SYCL exception will be thrown. A feature_not_supported
  /// exception will be thrown if any device that the program is to be linked
  /// for returns false for the device information query
  /// info::device::is_linker_available.
  ///
  /// @param ProgramList is a list of SYCL program instances.
  /// @param LinkOptions is a string containing valid OpenCL link options.
  program(vector_class<program> ProgramList, string_class LinkOptions = "");

  /// Constructs a SYCL program instance from an OpenCL cl_program.
  ///
  /// The state of the constructed SYCL program can be either
  /// program_state::compiled or program_state::linked, depending on the state
  /// of the ClProgram. Otherwise an invalid_object_error SYCL exception is
  /// thrown.
  ///
  /// The instance of OpenCL cl_program will be retained on construction.
  ///
  /// @param Context is an instance of SYCL Context.
  /// @param ClProgram is an instance of OpenCL cl_program.
  program(const context &Context, cl_program ClProgram);

  program(const program &rhs) = default;

  program(program &&rhs) = default;

  program &operator=(const program &rhs) = default;

  program &operator=(program &&rhs) = default;

  bool operator==(const program &rhs) const;

  bool operator!=(const program &rhs) const;

  /// Get a valid cl_program instance.
  ///
  /// The instance of cl_program will be retained before returning.
  /// If the program is created for a SYCL host device, an invalid_object_error
  /// exception is thrown.
  ///
  /// @return a valid OpenCL cl_program instance.
  cl_program get() const;

  /// Check if the program is created for a SYCL host device.
  ///
  /// @return true if this SYCL program is a host program.
  bool is_host() const;

  /// Compile the SYCL kernel function into the encapsulated raw program.
  ///
  /// The kernel function is defined by the type KernelT. This member function
  /// sets the state of this SYCL program to program_state::compiled.
  /// If this program was not in the program_state::none state,
  /// an invalid_object_error exception is thrown. If the compilation fails,
  /// a compile_program_error SYCL exception is thrown. If any device that the
  /// program is being compiled for returns false for the device information
  /// query info::device::is_compiler_available, a feature_not_supported
  /// exception is thrown.
  ///
  /// @param CompileOptions is a string of valid OpenCL compile options.
  template <typename KernelT>
  void compile_with_kernel_type(string_class CompileOptions = "") {
    detail::OSModuleHandle M = detail::OSUtil::getOSModuleHandle(
        detail::KernelInfo<KernelT>::getName());
    compile_with_kernel_type(detail::KernelInfo<KernelT>::getName(),
                             CompileOptions, M);
  }

  /// Compiles the OpenCL C kernel function defined by source string.
  ///
  /// This member function sets the state of this SYCL program to
  /// program_state::compiled.
  /// If the program was not in the program_state::none state,
  /// an invalid_object_error SYCL exception is thrown. If the compilation fails,
  /// a compile_program_error SYCL exception is thrown. If any device that the
  /// program is being compiled for returns false for the device information
  /// query info::device::is_compiler_available, a feature_not_supported
  /// SYCL exception is thrown.
  ///
  /// @param KernelSource is a string containing OpenCL C kernel source code.
  /// @param CompileOptions is a string containing OpenCL compile options.
  void compile_with_source(string_class KernelSource,
                           string_class CompileOptions = "");

  /// Builds the SYCL kernel function into encapsulated raw program.
  ///
  /// The SYCL kernel function is defined by the type KernelT.
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If the program was not in the program_state::none
  /// state, an invalid_object_error SYCL exception is thrown. If the compilation
  /// fails, a compile_program_error SYCL exception is thrown. If any device
  /// that the program is being built for returns false for the device
  /// information queries info::device::is_compiler_available or
  /// info::device::is_linker_available, a feature_not_supported SYCL exception
  /// is thrown.
  ///
  /// @param BuildOptions is a string containing OpenCL compile options.
  template <typename KernelT>
  void build_with_kernel_type(string_class BuildOptions = "") {
    detail::OSModuleHandle M = detail::OSUtil::getOSModuleHandle(
        detail::KernelInfo<KernelT>::getName());
    build_with_kernel_type(detail::KernelInfo<KernelT>::getName(), BuildOptions,
                           M);
  }

  /// Builds the OpenCL C kernel function defined by source code.
  ///
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If this program was not in program_state::none,
  /// an invalid_object_error SYCL exception is thrown. If the compilation fails,
  /// a compile_program_error SYCL exception is thrown. If any device
  /// that the program is being built for returns false for the device
  /// information queries info::device::is_compiler_available or
  /// info::device::is_linker_available, a feature_not_supported SYCL exception
  /// is thrown.
  ///
  /// @param KernelSource is a string containing OpenCL C kernel source code.
  /// @param BuildOptions is a string containing OpenCL build options.
  void build_with_source(string_class KernelSource,
                         string_class BuildOptions = "");

  /// Links encapsulated raw program.
  ///
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If the program was not in the program_state::compiled
  /// state, an invalid_object_error SYCL exception is thrown. If linking fails,
  /// a compile_program_error is thrown. If any device that the program is to be
  /// linked for returns false for the device information query
  /// info::device::is_linker_available, a feature_not_supported exception
  /// is thrown.
  ///
  /// @param LinkOptions is a string containing OpenCL link options.
  void link(string_class LinkOptions = "");

  /// Check if kernel is available for this program.
  ///
  /// The SYCL kernel is defined by type KernelT. If the program state is
  /// program_state::none an invalid_object_error SYCL exception is thrown.
  ///
  /// @return true if the SYCL kernel is available.
  template <typename KernelT> bool has_kernel() const {
    return has_kernel(detail::KernelInfo<KernelT>::getName(), /*IsCreatedFromSource*/ false);
  }

  /// Check if kernel is available for this program.
  ///
  /// The SYCL kernel is defined by its name. If the program is in the
  /// program_stateP::none state, an invalid_object_error SYCL exception
  /// is thrown.
  ///
  /// @param KernelName is a string containing kernel name.
  /// @return true if the SYCL kernel is available and the program is not a
  /// SYCL host program.
  bool has_kernel(string_class kernelName) const;

  template <typename KernelT> kernel get_kernel() const {
    return get_kernel(detail::KernelInfo<KernelT>::getName(),
                      /*IsCreatedFromSource*/ false);
  }

  kernel get_kernel(string_class kernelName) const;

  template <info::program param>
  typename info::param_traits<info::program, param>::return_type
  get_info() const;

  vector_class<vector_class<char>> get_binaries() const;

  context get_context() const;

  vector_class<device> get_devices() const;

  string_class get_compile_options() const;

  string_class get_link_options() const;

  string_class get_build_options() const;

  program_state get_state() const;

private:
  program(std::shared_ptr<detail::program_impl> impl);

  kernel get_kernel(string_class kernelName, bool IsCreatedFromSource) const;

  bool has_kernel(string_class kernelName, bool IsCreatedFromSource) const;

  void compile_with_kernel_type(string_class KernelName,
                                string_class compileOptions,
                                detail::OSModuleHandle M);

  void build_with_kernel_type(string_class KernelName,
                              string_class buildOptions,
                              detail::OSModuleHandle M);

  std::shared_ptr<detail::program_impl> impl;
};
} // namespace sycl
} // namespace cl

namespace std {
template <> struct hash<cl::sycl::program> {
  size_t operator()(const cl::sycl::program &prg) const {
    return hash<std::shared_ptr<cl::sycl::detail::program_impl>>()(
        cl::sycl::detail::getSyclObjImpl(prg));
  }
};
} // namespace std
