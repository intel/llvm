//==----- program_impl.hpp --- SYCL program implementation -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/context_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/spec_constant_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/kernel_desc.hpp>
#include <sycl/device.hpp>
#include <sycl/property_list.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>
#include <mutex>

namespace sycl {
inline namespace _V1 {

// Forward declarations
class kernel;

namespace detail {

using ContextImplPtr = std::shared_ptr<detail::context_impl>;

enum class program_state { none = 0, compiled = 1, linked = 2 };

class program_impl {
public:
  program_impl() = delete;

  /// Constructs an instance of program.
  ///
  /// The program will be created in the program_state::none state and
  /// associated with the provided context and the devices that are associated
  /// with the context.
  ///
  /// \param Context is a pointer to SYCL context impl.
  /// \param PropList is an instance of property_list.
  explicit program_impl(ContextImplPtr Context, const property_list &PropList);

  /// Constructs an instance of SYCL program for the provided DeviceList.
  ///
  /// The program will be created in the program_state::none state and
  /// associated with the provided context and the devices in the provided
  /// DeviceList.
  ///
  /// \param Context is a pointer to SYCL context impl.
  /// \param DeviceList is a list of SYCL devices.
  /// \param PropList is an instance of property_list.
  program_impl(ContextImplPtr Context, std::vector<device> DeviceList,
               const property_list &PropList);

  /// Constructs an instance of SYCL program by linking together each SYCL
  /// program instance in ProgramList.
  ///
  /// Each program in ProgramList must be in the program_state::compiled
  /// state and must be associated with the same SYCL context. Otherwise an
  /// invalid_object_error SYCL exception will be thrown. A
  /// feature_not_supported exception will be thrown if any device that the
  /// program is to be linked for returns false for the device information
  /// query info::device::is_linker_available. Kernels caching for linked
  /// programs won't be allowed due to only compiled state of each and every
  /// program in the list and thus unknown state of caching resolution.
  ///
  /// \param ProgramList is a list of program_impl instances.
  /// \param LinkOptions is a string containing valid OpenCL link options.
  /// \param PropList is an instance of property_list.
  program_impl(std::vector<std::shared_ptr<program_impl>> ProgramList,
               std::string LinkOptions, const property_list &PropList);

  /// Constructs a program instance from an interop raw BE program handle.
  /// TODO: BE generalization will change that to something better.
  ///
  /// The state of the constructed program can be either
  /// program_state::compiled or program_state::linked, depending on the state
  /// of the InteropProgram. Otherwise an invalid_object_error SYCL exception is
  /// thrown.
  ///
  /// The instance of the program will be retained on construction.
  ///
  /// \param Context is a pointer to SYCL context impl.
  /// \param InteropProgram is an instance of plugin interface interoperability
  /// program.
  program_impl(ContextImplPtr Context, pi_native_handle InteropProgram);

  /// Constructs a program instance from plugin interface interoperability
  /// kernel.
  ///
  /// \param Context is a pointer to SYCL context impl.
  /// \param Kernel is a raw PI kernel handle.
  program_impl(ContextImplPtr Context, sycl::detail::pi::PiKernel Kernel);

  ~program_impl();

  /// Checks if this program_impl has a property of type propertyT.
  ///
  /// \return true if this program_impl has a property of type propertyT.
  template <typename propertyT> bool has_property() const {
    return MPropList.has_property<propertyT>();
  }

  /// Gets the specified property of this program_impl.
  ///
  /// Throws invalid_object_error if this program_impl does not have a property
  /// of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const {
    return MPropList.get_property<propertyT>();
  }

  /// Returns a valid cl_program instance.
  ///
  /// The instance of cl_program will be retained before returning.
  /// If the program is created for a SYCL host device, an
  /// invalid_object_error exception is thrown.
  ///
  /// \return a valid OpenCL cl_program instance.
  cl_program get() const;

  /// \return a reference to a raw PI program handle. PI program is not
  /// retained before return.
  sycl::detail::pi::PiProgram &getHandleRef() { return MProgram; }
  /// \return a constant reference to a raw PI program handle. PI program is
  /// not retained before return.
  const sycl::detail::pi::PiProgram &getHandleRef() const { return MProgram; }

  /// Compiles the SYCL kernel function into the encapsulated raw program.
  ///
  /// The kernel function is defined by its name. This member function
  /// sets the state of this SYCL program to program_state::compiled.
  /// If this program was not in the program_state::none state,
  /// an invalid_object_error exception is thrown. If the compilation fails,
  /// a compile_program_error SYCL exception is thrown. If any device that the
  /// program is being compiled for returns false for the device information
  /// query info::device::is_compiler_available, a feature_not_supported
  /// exception is thrown.
  ///
  /// \param KernelName is a string containing SYCL kernel name.
  /// \param CompileOptions is a string of valid OpenCL compile options.
  /// \param Module is an OS handle to user code module.
  void compile_with_kernel_name(std::string KernelName,
                                std::string CompileOptions);

  /// Builds the SYCL kernel function into encapsulated raw program.
  ///
  /// The SYCL kernel function is defined by the kernel name.
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If the program was not in the program_state::none
  /// state, an invalid_object_error SYCL exception is thrown. If the
  /// compilation fails, a compile_program_error SYCL exception is thrown. If
  /// any device that the program is being built for returns false for the
  /// device information queries info::device::is_compiler_available or
  /// info::device::is_linker_available, a feature_not_supported SYCL
  /// exception is thrown.
  ///
  /// \param KernelName is a string containing SYCL kernel name.
  /// \param BuildOptions is a string containing OpenCL compile options.
  /// \param M is an OS handle to user code module.
  void build_with_kernel_name(std::string KernelName, std::string BuildOptions);

  /// Links encapsulated raw program.
  ///
  /// This member function sets the state of this SYCL program to
  /// program_state::linked. If the program was not in the
  /// program_state::compiled state, an invalid_object_error SYCL exception is
  /// thrown. If linking fails, a compile_program_error is thrown. If any
  /// device that the program is to be linked for returns false for the device
  /// information query info::device::is_linker_available, a
  /// feature_not_supported exception is thrown.
  ///
  /// \param LinkOptions is a string containing OpenCL link options.
  void link(std::string LinkOptions = "");

  /// Checks if kernel is available for this program.
  ///
  /// The SYCL kernel is defined by kernel name. If the program state is
  /// program_state::none an invalid_object_error SYCL exception is thrown.
  ///
  /// \return true if the SYCL kernel is available.
  bool has_kernel(std::string KernelName, bool IsCreatedFromSource) const;

  /// Returns a SYCL kernel for the SYCL kernel function defined by kernel
  /// name.
  ///
  /// If program is in the program_state::none state or if the SYCL kernel
  /// function is not available, an invalid_object_error exception is thrown.
  ///
  /// \return a valid instance of SYCL kernel.
  kernel get_kernel(std::string KernelName,
                    std::shared_ptr<program_impl> PtrToSelf,
                    bool IsCreatedFromSource) const;

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
  context get_context() const {
    return createSyclObjFromImpl<context>(MContext);
  }

  /// \return the Plugin associated with the context of this program.
  const PluginPtr &getPlugin() const { return MContext->getPlugin(); }

  ContextImplPtr getContextImplPtr() const { return MContext; }

  /// \return a vector of devices that are associated with this program.
  std::vector<device> get_devices() const { return MDevices; }

  /// Returns compile options that were provided when the encapsulated program
  /// was explicitly compiled.
  ///
  /// If the program was built instead of explicitly compiled, if the program
  /// has not yet been compiled, or if the program has been compiled for only
  /// the host device, then an empty string is return, unless the underlying
  /// cl_program was explicitly compiled, in which case the compile options
  /// used in the explicit compile are returned.
  ///
  /// \return a string of valid OpenCL compile options.
  std::string get_compile_options() const { return MCompileOptions; }

  /// Returns compile options that were provided to the most recent invocation
  /// of link member function.
  ///
  /// If the program has not been explicitly linked using the aforementioned
  /// function, constructed with an explicitly linking constructor, or if the
  /// program has been linked for only the host device, then an empty string
  /// is returned. If the program was constructed from cl_program, then an
  /// empty string is returned unless the cl_program was explicitly linked,
  /// in which case the link options used in that explicit link are returned.
  /// If the program object was constructed using a constructor form that
  /// links a vector of programs, then the link options passed to this
  /// constructor are returned.
  ///
  /// \return a string of valid OpenCL compile options.
  std::string get_link_options() const { return MLinkOptions; }

  /// Returns the compile, link, or build options, from whichever of those
  /// operations was performed most recently on the encapsulated cl_program.
  ///
  /// If no compile, link, or build operations have been performed on this
  /// program, or if the program includes the host device in its device list,
  /// then an empty string is returned.
  ///
  /// \return a string of valid OpenCL build options.
  std::string get_build_options() const { return MBuildOptions; }

  /// \return the current state of this SYCL program.
  program_state get_state() const { return MState; }

  /// Takes current values of specialization constants and "injects" them into
  /// the underlying native program program via specialization constant
  /// managemment PI APIs. The native program passed as non-null argument
  /// overrides the MProgram native program field.
  /// \param Img device binary image corresponding to this program, used to
  ///        resolve spec constant name to SPIR-V integer ID
  /// \param NativePrg if not null, used as the flush target, otherwise MProgram
  ///        is used
  void
  flush_spec_constants(const RTDeviceBinaryImage &Img,
                       sycl::detail::pi::PiProgram NativePrg = nullptr) const;

  void stableSerializeSpecConstRegistry(SerializedObj &Dst) const {
    detail::stableSerializeSpecConstRegistry(SpecConstRegistry, Dst);
  }

  /// Tells whether a specialization constant has been set for this program.
  bool hasSetSpecConstants() const { return !SpecConstRegistry.empty(); }

  /// \return true if caching is allowed for this program.
  bool is_cacheable() const { return MProgramAndKernelCachingAllowed; }

  /// Returns the native plugin handle.
  pi_native_handle getNative() const;

  bool isInterop() const { return MIsInterop; }

private:
  // Deligating Constructor used in Implementation.
  program_impl(ContextImplPtr Context, pi_native_handle InteropProgram,
               sycl::detail::pi::PiProgram Program);
  /// Checks feature support for specific devices.
  ///
  /// If there's at least one device that does not support this feature,
  /// a feature_not_supported exception is thrown.
  ///
  /// \param Devices is a vector of SYCL devices.
  template <typename Param>
  void check_device_feature_support(const std::vector<device> &Devices) {
    for (const auto &Device : Devices) {
      if (!Device.get_info<Param>()) {
        throw sycl::exception(
            sycl::errc::feature_not_supported,
            "Online compilation is not supported by this device");
      }
    }
  }

  /// Creates a plugin interface kernel using its name.
  ///
  /// \param Module is an OS handle to user code module.
  /// \param KernelName is a name of kernel to be created.
  /// \param JITCompilationIsRequired If JITCompilationIsRequired is true
  ///        add a check that kernel is compiled, otherwise don't add the check.
  void
  create_pi_program_with_kernel_name(const std::string &KernelName,
                                     bool JITCompilationIsRequired = false);

  /// Compiles underlying plugin interface program.
  ///
  /// \param Options is a string containing OpenCL compile options.
  void compile(const std::string &Options);

  /// Builds underlying plugin interface program.
  ///
  /// \param Options is a string containing OpenCL build options.
  void build(const std::string &Options);

  /// \return a vector of devices managed by the plugin.
  std::vector<sycl::detail::pi::PiDevice> get_pi_devices() const;

  /// \param Options is a string containing OpenCL C build options.
  /// \return true if caching is allowed for this program and build options.
  static bool is_cacheable_with_options(const std::string &Options) {
    return Options.empty();
  }

  /// \param KernelName is a string containing OpenCL kernel name.
  /// \return true if underlying OpenCL program has kernel with specific name.
  bool has_cl_kernel(const std::string &KernelName) const;

  /// \param KernelName is a string containing PI kernel name.
  /// \return an instance of PI kernel with specific name. If kernel is
  /// unavailable, an invalid_object_error exception is thrown.
  std::pair<sycl::detail::pi::PiKernel, const KernelArgMask *>
  get_pi_kernel_arg_mask_pair(const std::string &KernelName) const;

  /// \return a vector of sorted in ascending order SYCL devices.
  std::vector<device> sort_devices_by_cl_device_id(std::vector<device> Devices);

  /// Throws an invalid_object_exception if state of this program is in the
  /// specified state.
  ///
  /// \param State is a program state to match against.
  void throw_if_state_is(program_state State) const;

  /// Throws an invalid_object_exception if state of this program is not in
  /// the specified state.
  ///
  /// \param State is a program state to match against.
  void throw_if_state_is_not(program_state State) const;

  sycl::detail::pi::PiProgram MProgram = nullptr;
  program_state MState = program_state::none;
  std::mutex MMutex;
  ContextImplPtr MContext;
  bool MLinkable = false;
  std::vector<device> MDevices;
  property_list MPropList;
  std::string MCompileOptions;
  std::string MLinkOptions;
  std::string MBuildOptions;

  // Keeps specialization constant map for this program. Spec constant name
  // resolution to actual SPIR-V integer ID happens at build time, where the
  // device binary image is available. Access is guarded by this context's
  // program cache lock.
  SpecConstRegistryT SpecConstRegistry;

  /// Only allow kernel caching for programs constructed with context only (or
  /// device list and context) and built with build_with_kernel_type with
  /// default build options
  bool MProgramAndKernelCachingAllowed = false;

  bool MIsInterop = false;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
