//==--------------- program.hpp --- SYCL program ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <sycl/ext/oneapi/experimental/spec_constant.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations
class context;
class device;
namespace detail {
class program_impl;
}

enum class program_state { none = 0, compiled = 1, linked = 2 };

/// Provides an abstraction of a SYCL program.
///
/// \sa kernel
/// \sa queue
///
/// \ingroup sycl_api
class __SYCL_EXPORT __SYCL2020_DEPRECATED(
    "program class is deprecated, use kernel_bundle instead") program {
public:
  program() = delete;

  /// Constructs an instance of SYCL program.
  ///
  /// The program will be created in the program_state::none state and
  /// associated with the provided context and the SYCL devices that are
  /// associated with the context.
  ///
  /// \param Context is an instance of SYCL context.
  /// \param PropList is an instance of property_list.
  explicit program(const context &Context, const property_list &PropList = {});

  /// Constructs an instance of SYCL program for the provided DeviceList.
  ///
  /// The program will be created in the program_state::none state and
  /// associated with the provided context and the SYCL devices in the provided
  /// DeviceList.
  ///
  /// \param Context is an instance of SYCL context.
  /// \param DeviceList is a list of SYCL devices.
  /// \param PropList is an instance of property_list.
  program(const context &Context, std::vector<device> DeviceList,
          const property_list &PropList = {});

  /// Constructs an instance of SYCL program by linking together each SYCL
  /// program instance in ProgramList.
  ///
  /// Each SYCL program in ProgramList must be in the program_state::compiled
  /// state and must be associated with the same SYCL context. Otherwise an
  /// invalid_object_error SYCL exception will be thrown. A
  /// feature_not_supported exception will be thrown if any device that the
  /// program is to be linked for returns false for the device information query
  /// info::device::is_linker_available.
  ///
  /// \param ProgramList is a list of SYCL program instances.
  /// \param PropList is an instance of property_list.
  program(std::vector<program> ProgramList, const property_list &PropList = {});

  /// Constructs an instance of SYCL program by linking together each SYCL
  /// program instance in ProgramList.
  ///
  /// Each SYCL program in ProgramList must be in the program_state::compiled
  /// state and must be associated with the same SYCL context. Otherwise an
  /// invalid_object_error SYCL exception will be thrown. A
  /// feature_not_supported exception will be thrown if any device that the
  /// program is to be linked for returns false for the device information query
  /// info::device::is_linker_available.
  ///
  /// \param ProgramList is a list of SYCL program instances.
  /// \param LinkOptions is a string containing valid OpenCL link options.
  /// \param PropList is an instance of property_list.
  program(std::vector<program> ProgramList, std::string LinkOptions,
          const property_list &PropList = {});

  /// Constructs a SYCL program instance from an OpenCL cl_program.
  ///
  /// The state of the constructed SYCL program can be either
  /// program_state::compiled or program_state::linked, depending on the state
  /// of the ClProgram. Otherwise an invalid_object_error SYCL exception is
  /// thrown.
  ///
  /// The instance of OpenCL cl_program will be retained on construction.
  ///
  /// \param Context is an instance of SYCL Context.
  /// \param ClProgram is an instance of OpenCL cl_program.
  program(const context &Context, cl_program ClProgram);

  program(const program &rhs) = default;

  program(program &&rhs) = default;

  program &operator=(const program &rhs) = default;

  program &operator=(program &&rhs) = default;

  bool operator==(const program &rhs) const { return impl == rhs.impl; }

  bool operator!=(const program &rhs) const { return impl != rhs.impl; }

  /// Checks if this program has a property of type propertyT.
  ///
  /// \return true if this context has a property of type propertyT.
  template <typename propertyT> bool has_property() const;

  /// Gets the specified property of this program.
  ///
  /// Throws invalid_object_error if this program does not have a property
  /// of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const;

  /// Returns a valid cl_program instance.
  ///
  /// The instance of cl_program will be retained before returning.
  /// If the program is created for a SYCL host device, an invalid_object_error
  /// exception is thrown.
  ///
  /// \return a valid OpenCL cl_program instance.
  cl_program get() const;

  /// \return true if this SYCL program is a host program.
  bool is_host() const;

  /// Compiles the SYCL kernel function into the encapsulated raw program.
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
  /// \param CompileOptions is a string of valid OpenCL compile options.
  template <typename KernelT>
  void compile_with_kernel_type(std::string CompileOptions = "") {
    detail::OSModuleHandle M = detail::OSUtil::getOSModuleHandle(
        detail::KernelInfo<KernelT>::getName());
    compile_with_kernel_name(detail::KernelInfo<KernelT>::getName(),
                             CompileOptions, M);
  }

  /// Compiles the OpenCL C kernel function defined by source string.
  ///
  /// This member function sets the state of this SYCL program to
  /// program_state::compiled.
  /// If the program was not in the program_state::none state,
  /// an invalid_object_error SYCL exception is thrown. If the compilation
  /// fails, a compile_program_error SYCL exception is thrown. If any device
  /// that the program is being compiled for returns false for the device
  /// information query info::device::is_compiler_available, a
  /// feature_not_supported SYCL exception is thrown.
  ///
  /// \param KernelSource is a string containing OpenCL C kernel source code.
  /// \param CompileOptions is a string containing OpenCL compile options.
  void compile_with_source(std::string KernelSource,
                           std::string CompileOptions = "");

  /// Builds the SYCL kernel function into encapsulated raw program.
  ///
  /// The SYCL kernel function is defined by the type KernelT.
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If the program was not in the program_state::none
  /// state, an invalid_object_error SYCL exception is thrown. If the
  /// compilation fails, a compile_program_error SYCL exception is thrown. If
  /// any device that the program is being built for returns false for the
  /// device information queries info::device::is_compiler_available or
  /// info::device::is_linker_available, a feature_not_supported SYCL exception
  /// is thrown.
  ///
  /// \param BuildOptions is a string containing OpenCL compile options.
  template <typename KernelT>
  void build_with_kernel_type(std::string BuildOptions = "") {
    detail::OSModuleHandle M = detail::OSUtil::getOSModuleHandle(
        detail::KernelInfo<KernelT>::getName());
    build_with_kernel_name(detail::KernelInfo<KernelT>::getName(), BuildOptions,
                           M);
  }

  /// Builds the OpenCL C kernel function defined by source code.
  ///
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If this program was not in program_state::none,
  /// an invalid_object_error SYCL exception is thrown. If the compilation
  /// fails, a compile_program_error SYCL exception is thrown. If any device
  /// that the program is being built for returns false for the device
  /// information queries info::device::is_compiler_available or
  /// info::device::is_linker_available, a feature_not_supported SYCL exception
  /// is thrown.
  ///
  /// \param KernelSource is a string containing OpenCL C kernel source code.
  /// \param BuildOptions is a string containing OpenCL build options.
  void build_with_source(std::string KernelSource,
                         std::string BuildOptions = "");

  /// Links encapsulated raw program.
  ///
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If the program was not in the
  /// program_state::compiled state, an invalid_object_error SYCL exception is
  /// thrown. If linking fails, a compile_program_error is thrown. If any device
  /// that the program is to be linked for returns false for the device
  /// information query info::device::is_linker_available, a
  /// feature_not_supported exception is thrown.
  ///
  /// \param LinkOptions is a string containing OpenCL link options.
  void link(std::string LinkOptions = "");

  /// Checks if kernel is available for this program.
  ///
  /// The SYCL kernel is defined by type KernelT. If the program state is
  /// program_state::none an invalid_object_error SYCL exception is thrown.
  ///
  /// \return true if the SYCL kernel is available.
  template <typename KernelT> bool has_kernel() const {
    return has_kernel(detail::KernelInfo<KernelT>::getName(),
                      /*IsCreatedFromSource*/ false);
  }

  /// Checks if kernel is available for this program.
  ///
  /// The SYCL kernel is defined by its name. If the program is in the
  /// program_stateP::none state, an invalid_object_error SYCL exception
  /// is thrown.
  ///
  /// \param KernelName is a string containing kernel name.
  /// \return true if the SYCL kernel is available and the program is not a
  /// SYCL host program.
  bool has_kernel(std::string KernelName) const;

  /// Returns a SYCL kernel for the SYCL kernel function defined by KernelType.
  ///
  /// If program is in the program_state::none state or if the SYCL kernel
  /// function is not available, an invalid_object_error exception is thrown.
  ///
  /// \return a valid instance of SYCL kernel.
  template <typename KernelT> kernel get_kernel() const {
    return get_kernel(detail::KernelInfo<KernelT>::getName(),
                      /*IsCreatedFromSource*/ false);
  }

  /// Returns a SYCL kernel for the SYCL kernel function defined by KernelName.
  ///
  /// An invalid_object_error SYCL exception is thrown if this program is a host
  /// program, if program is in the program_state::none state or if the SYCL
  /// kernel is not available.
  ///
  /// \param KernelName is a string containing SYCL kernel name.
  kernel get_kernel(std::string KernelName) const;

  /// Queries this SYCL program for information.
  ///
  /// The return type depends on the information being queried.
  template <info::program param>
  typename info::param_traits<info::program, param>::return_type
  get_info() const;

  /// Returns built program binaries.
  ///
  /// If this program is not in the program_state::compiled or
  /// program_state::linked states, an invalid_object_error SYCL exception
  /// is thrown.
  ///
  /// \return a vector of vectors representing the compiled binaries for each
  /// associated SYCL device.
  std::vector<std::vector<char>> get_binaries() const;

  /// \return the SYCL context that this program was constructed with.
  context get_context() const;

  /// \return a vector of devices that are associated with this program.
  std::vector<device> get_devices() const;

  /// Returns compile options that were provided when the encapsulated program
  /// was explicitly compiled.
  ///
  /// If the program was built instead of explicitly compiled, if the program
  /// has not yet been compiled, or if the program has been compiled for only
  /// the host device, then an empty string is return, unless the underlying
  /// cl_program was explicitly compiled, in which case the compile options used
  /// in the explicit compile are returned.
  ///
  /// \return a string of valid OpenCL compile options.
  std::string get_compile_options() const;

  /// Returns compile options that were provided to the most recent invocation
  /// of link member function.
  ///
  /// If the program has not been explicitly linked using the aforementioned
  /// function, constructed with an explicitly linking constructor, or if the
  /// program has been linked for only the host device, then an empty string
  /// is returned. If the program was constructed from cl_program, then an
  /// empty string is returned unless the cl_program was explicitly linked,
  /// in which case the link options used in that explicit link are returned.
  /// If the program object was constructed using a constructor form that links
  /// a vector of programs, then the link options passed to this constructor
  /// are returned.
  ///
  /// \return a string of valid OpenCL compile options.
  std::string get_link_options() const;

  /// Returns the compile, link, or build options, from whichever of those
  /// operations was performed most recently on the encapsulated cl_program.
  ///
  /// If no compile, link, or build operations have been performed on this
  /// program, or if the program includes the host device in its device list,
  /// then an empty string is returned.
  ///
  /// \return a string of valid OpenCL build options.
  std::string get_build_options() const;

  /// \return the current state of this SYCL program.
  program_state get_state() const;

  /// Set the value of the specialization constant identified by the 'ID' type
  /// template parameter and return its instance.
  /// \param cst the specialization constant value
  /// \return a specialization constant instance corresponding to given type ID
  ///         passed as a template parameter
  template <typename ID, typename T>
  ext::oneapi::experimental::spec_constant<T, ID> set_spec_constant(T Cst) {
    constexpr const char *Name = detail::SpecConstantInfo<ID>::getName();
    static_assert(std::is_arithmetic<T>::value ||
                      (std::is_class<T>::value && std::is_pod<T>::value),
                  "unsupported specialization constant type");
#ifdef __SYCL_DEVICE_ONLY__
    (void)Cst;
    (void)Name;
    return ext::oneapi::experimental::spec_constant<T, ID>();
#else
    set_spec_constant_impl(Name, &Cst, sizeof(T));
    return ext::oneapi::experimental::spec_constant<T, ID>(Cst);
#endif // __SYCL_DEVICE_ONLY__
  }

  /// Returns the backend associated with this program.
  ///
  /// \return the backend associated with this program.
  backend get_backend() const noexcept;

  /// Gets the native handle of the SYCL platform.
  ///
  /// \return a native handle, the type of which defined by the backend.
  template <backend BackendName>
  auto get_native() const -> typename interop<BackendName, program>::type {
    return reinterpret_cast<typename interop<BackendName, program>::type>(
        getNative());
  }

private:
  pi_native_handle getNative() const;
  program(std::shared_ptr<detail::program_impl> impl);

  /// Template-free version of get_kernel.
  ///
  /// \param KernelName is a stringified kernel name.
  /// \param IsCreatedFromSource is a flag indicating whether this program was
  /// created from OpenCL C source code string.
  /// \return a valid instance of SYCL kernel.
  kernel get_kernel(std::string KernelName, bool IsCreatedFromSource) const;

  /// Template-free version of has_kernel.
  ///
  /// \param KernelName is a stringified kernel name.
  /// \param IsCreatedFromSource is a flag indicating whether this program was
  /// created from OpenCL C source code string.
  /// \return true if kernel with KernelName is available.
  bool has_kernel(std::string KernelName, bool IsCreatedFromSource) const;

  /// Template-free version of compile_with_kernel_type.
  ///
  /// \param KernelName is a stringified kernel name.
  /// \param CompileOptions is a string of valid OpenCL compile options.
  /// \param M is a valid OS handle to the user executable or library.
  void compile_with_kernel_name(std::string KernelName,
                                std::string CompileOptions,
                                detail::OSModuleHandle M);

  /// Template-free version of build_with_kernel_type.
  ///
  /// \param KernelName is a stringified kernel name.
  /// \param CompileOptions is a string of valid OpenCL compile options.
  /// \param M is a valid OS handle to the user executable or library.
  void build_with_kernel_name(std::string KernelName, std::string buildOptions,
                              detail::OSModuleHandle M);

  void set_spec_constant_impl(const char *Name, void *Data, size_t Size);

  std::shared_ptr<detail::program_impl> impl;

  template <class Obj>
  friend decltype(Obj::impl) detail::getSyclObjImpl(const Obj &SyclObject);
  template <class T>
  friend T detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);
};
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace std {
template <> struct hash<cl::sycl::program> {
  size_t operator()(const cl::sycl::program &prg) const {
    return hash<std::shared_ptr<cl::sycl::detail::program_impl>>()(
        cl::sycl::detail::getSyclObjImpl(prg));
  }
};
} // namespace std
