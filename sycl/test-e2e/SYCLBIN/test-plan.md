# Test plan for `sycl_ext_oneapi_syclbin`

Spec: <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_syclbin.asciidoc>  

Design docs:

- <https://github.com/intel/llvm/blob/sycl/sycl/doc/design/SYCLBINDesign.md>
- <https://github.com/intel/llvm/blob/sycl/sycl/doc/design/PropertySets.md>

Tests:

 1. Tests for

    ```
    namespace sycl::ext::oneapi::experimental {

    template<bundle_state State, typename PropertyListT = empty_properties_t>
    kernel_bundle<State> get_kernel_bundle(const context& ctxt,
                                        const std::vector<device>& devs,
                                        const sycl::span<char>& bytes,
                                        PropertyListT props = {});

    template<bundle_state State, typename PropertyListT = empty_properties_t>
    kernel_bundle<State> get_kernel_bundle(const context& ctxt,
                                        const std::vector<device>& devs,
                                        const std::span<char>& bytes,
                                        PropertyListT props = {});

    }
    ```

    1. Check APIs with all available  `bundle_state`s, except 
       `bundle_state::ext_oneapi_source`
    2. Validate APIs produce valid `kernel_bundle<State>` on discrete Intel GPU
       and on Intel CPU (e.g., its state must be `State`, etc.)
    3. Validate that every kernel in returned `kernel_bundle<State>` is
       compatible with at least one device from `devs`
    4. Check an exception with the `errc::invalid` error code is thrown if the
       contents of bytes is not in the SYCLBIN format
    5. Check an exception with the `errc::invalid` error code is thrown if the
       SYCLBIN read from bytes is not in the `State` state
    6. Check an exception with the `errc::invalid` error code is thrown if the
       devs vector is empty
    7. Check an exception with the `errc::invalid` error code is thrown if
       `State` is `bundle_state::input` and any device in `ctxt.get_devices()`
       does not have `aspect::online_compiler` (unit test)
    8. Check an exception with the `errc::invalid` error code is thrown if
       `State` is `bundle_state::object` and any device in `ctxt.get_devices()`
       does not have `aspect::online_linker` (unit test)
    9. Check an exception with the `errc::build` error code is thrown if
       `State` is `bundle_state::object` or `bundle_state::executable`, if the
       implementation needs to perform an online compile, and if the online
       compile fails (unit test)
    10. Check an exception with the `errc::build` error code is thrown if
       `State` is `bundle_state::object` or `bundle_state::executable`, if the
        implementation needs to perform a link, and if the link fails (unit
        test)

 2. Tests for

    ```
    namespace sycl::ext::oneapi::experimental {

    template<bundle_state State, typename PropertyListT = empty_properties_t>      (1)
    kernel_bundle<State> get_kernel_bundle(const context& ctxt,
                                        const std::vector<device>& devs,
                                        const std::filesystem::path& filename,
                                         PropertyListT props = {});

    template<bundle_state State, typename PropertyListT = empty_properties_t>      (2)
    kernel_bundle<State> get_kernel_bundle(const context& ctxt,
                                        const std::filesystem::path& filename,
                                        PropertyListT props = {});

    }
    ```

    1. Check APIs with all available  `bundle_state`s, except
       `bundle_state::ext_oneapi_source`
    2. Validate APIs produce valid `kernel_bundle<State>` on discrete Intel GPU
       and on Intel CPU (e.g., its state must be `State`, etc.)
    3. Validate that every kernel in returned `kernel_bundle<State>` is
       compatible with at least one device from `devs` for (1)
    4. Check a `std::ios_base::failure` exception is thrown if the function
       failed to access and read the file specified by filename (unit test)
    5. Check an exception with the `errc::invalid` error code is thrown if the
       contents of the file specified by filename is not in the SYCLBIN format
    6. Check an exception with the `errc::invalid` error code is thrown if the
       SYCLBIN read from the file specified by filename is not in the `State`
       state
    7. Check an exception with the `errc::invalid` error code is thrown if any
       of the devices in `devs` is not one of devices contained by the context
       `ctxt` or is not a descendent device of some device in `ctxt` (unit test)
    8. Check an exception with the `errc::invalid` error code is thrown if the
       `devs` vector is empty
    9. Check an exception with the `errc::invalid` error code is thrown if
       `State` is `bundle_state::input` and any device in `ctxt.get_devices()`
       does not have `aspect::online_compiler` (unit test)
    10. Check an exception with the `errc::invalid` error code is thrown if
       `State` is `bundle_state::object` and any device in `ctxt.get_devices()`
       does not have `aspect::online_linker` (unit test)

 3. Tests for

    ```
    namespace sycl {
    template <bundle_state State> class kernel_bundle {
    public:
    ...

    std::vector<char> ext_oneapi_get_content();

    };
    }
    ```

    1. Check API with all available  `bundle_state`s, except
       `bundle_state::ext_oneapi_source`
    2. Check the returned `std::vector<char>` contains the expected data of the
       kernel bundle in the SYCLBIN format.
    3. Check the corresponding SYCLBIN format in the returned
       `std::vector<char>` has `State` state

 4. Tests for Clang driver

    1. Check error generated if use `-fsyclbin` without `--offload-new-driver`
    2. Check when `-fsyclbin` is set, host-compilation invocation of `-fsycl`
       pipeline is skipped
    3. Check when `-fsyclbin` is set, output is a file with the .syclbin file
       extension
    4. Check when `-fsyclbin` is set, `clang-linker-wrapper` has `--syclbin`
       flag
    5. Check when `-fsyclbin` is not set, `clang-linker-wrapper` has `--syclbin`
       flag
    6. Check when `-fsycl-device-only` and `-fsyclbin` are used,
       `-fsycl-device-only` is unused
    7. Check `-fsycl-link` works without any errors with .syclbin files. TODO:
       figure out that
    8. Check `--offload-rdc` is an alias to `-fgpu-rdc`

 5. Tests for clang-linker-wrapper

    1. Check when `--syclbin` is used, module-splitting is performed
    2. Check when `--syclbin` is used, `clang-linker-wrapper` skips wrapping of
       the device code and the host code linking stage

 6. Tests for SYCL runtime library

    1. Validate SYCL RT correctly parses SYCLBIN file (e.g., magic number is
       correct, etc.). Check correctness of corresponding data structure. Test
       should expect that SYCLBIN version is 1.
    2. Validate SYCL RT correctly writes SYCLBIN object to SYCLBIN file. Test
       should expect that SYCLBIN version is 1.
    3. Test SYCL RT correctly parses input SYCLBIN properies in binary format:
       SYCLBIN/global metadata, SYCLBIN/ir module metadata, SYCLBIN/native
       device code image metadata
