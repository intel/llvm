set(obj_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if (WIN32)
  set(spv_binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  set(libcrt_source msvc_wrapper.cpp)
else()
  set(spv_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(libcrt_source glibc_wrapper.cpp)
endif()
set(clang $<TARGET_FILE:clang>)

set(compile_opts
  # suppress an error about SYCL_EXTERNAL being used for
  # a function with a raw pointer parameter.
  -Wno-sycl-strict
  # Disable warnings for the host compilation, where
  # we declare all functions as 'static'.
  -Wno-undefined-internal
  # Force definition of CL_SYCL_LANGUAGE_VERSION, as long as
  # SYCL specific code is guarded by it.
  -sycl-std=2017
  )

function(BuildSYCLDeviceLib)
  cmake_parse_arguments(DeviceLib "" "Path;Target;Source" "Depends" ${ARGN})
  add_custom_command(OUTPUT ${DeviceLib_Path}
                     COMMAND ${clang} -fsycl -fsycl-device-only -emit-llvm
                             ${compile_opts} -fsycl-targets=${DeviceLib_Target}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${DeviceLib_Source}
                             -o ${DeviceLib_Path} -fno-sycl-device-lib=all
                     MAIN_DEPENDENCY ${DeviceLib_Source}
                     DEPENDS ${DeviceLib_Depends}
                     VERBATIM)
endfunction()


set(devicelib-wrapper-crt-spir64 ${obj_binary_dir}/libsycl-crt-spir64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-crt-spir64} Target spir64-unknown-unknown-sycldevice Source ${libcrt_source} Depends wrapper.h device.h spirv_vars.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cassert-spir64.bc Target spir64-unknown-unknown-sycldevice Source fallback-cassert.cpp Depends wrapper.h device.h spirv_vars.h clang)

set(devicelib-wrapper-crt-spir64_x86_64 ${obj_binary_dir}/libsycl-crt-spir64_x86_64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-crt-spir64_x86_64} Target spir64_x86_64-unknown-unknown-sycldevice Source ${libcrt_source} Depends wrapper.h device.h spirv_vars.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cassert-spir64_x86_64.bc Target spir64_x86_64-unknown-unknown-sycldevice Source fallback-cassert.cpp Depends wrapper.h device.h spirv_vars.h clang)

set(devicelib-wrapper-crt-spir64_gen ${obj_binary_dir}/libsycl-crt-spir64_gen.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-crt-spir64_gen} Target spir64_gen-unknown-unknown-sycldevice Source ${libcrt_source} Depends wrapper.h device.h spirv_vars.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cassert-spir64_gen.bc Target spir64_gen-unknown-unknown-sycldevice Source fallback-cassert.cpp Depends wrapper.h device.h spirv_vars.h clang)

set(devicelib-wrapper-crt-spir64_fpga ${obj_binary_dir}/libsycl-crt-spir64_fpga.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-crt-spir64_fpga} Target spir64_fpga-unknown-unknown-sycldevice Source ${libcrt_source} Depends wrapper.h device.h spirv_vars.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cassert-spir64_fpga.bc Target spir64_fpga-unknown-unknown-sycldevice Source fallback-cassert.cpp Depends wrapper.h device.h spirv_vars.h clang)

set(devicelib-wrapper-complex-spir64 ${obj_binary_dir}/libsycl-complex-spir64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-spir64} Target spir64-unknown-unknown-sycldevice Source complex_wrapper.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-spir64.bc Target spir64-unknown-unknown-sycldevice Source fallback-complex.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-complex-spir64_x86_64 ${obj_binary_dir}/libsycl-complex-spir64_x86_64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-spir64_x86_64} Target spir64_x86_64-unknown-unknown-sycldevice Source complex_wrapper.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-spir64_x86_64.bc Target spir64_x86_64-unknown-unknown-sycldevice Source fallback-complex.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-complex-spir64_gen ${obj_binary_dir}/libsycl-complex-spir64_gen.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-spir64_gen} Target spir64_gen-unknown-unknown-sycldevice Source complex_wrapper.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-spir64_gen.bc Target spir64_gen-unknown-unknown-sycldevice Source fallback-complex.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-complex-spir64_fpga ${obj_binary_dir}/libsycl-complex-spir64_fpga.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-spir64_fpga} Target spir64_fpga-unknown-unknown-sycldevice Source complex_wrapper.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-spir64_fpga.bc Target spir64_fpga-unknown-unknown-sycldevice Source fallback-complex.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-complex-fp64-spir64 ${obj_binary_dir}/libsycl-complex-fp64-spir64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-fp64-spir64} Target spir64-unknown-unknown-sycldevice Source complex_wrapper_fp64.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64.bc Target spir64-unknown-unknown-sycldevice Source fallback-complex-fp64.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-complex-fp64-spir64_x86_64 ${obj_binary_dir}/libsycl-complex-fp64-spir64_x86_64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-fp64-spir64_x86_64} Target spir64_x86_64-unknown-unknown-sycldevice Source complex_wrapper_fp64.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_x86_64.bc Target spir64_x86_64-unknown-unknown-sycldevice Source fallback-complex-fp64.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-complex-fp64-spir64_gen ${obj_binary_dir}/libsycl-complex-fp64-spir64_gen.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-fp64-spir64_gen} Target spir64_gen-unknown-unknown-sycldevice Source complex_wrapper_fp64.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_gen.bc Target spir64_gen-unknown-unknown-sycldevice Source fallback-complex-fp64.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-complex-fp64-spir64_fpga ${obj_binary_dir}/libsycl-complex-fp64-spir64_fpga.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-complex-fp64-spir64_fpga} Target spir64_fpga-unknown-unknown-sycldevice Source complex_wrapper_fp64.cpp Depends device_complex.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_fpga.bc Target spir64_fpga-unknown-unknown-sycldevice Source fallback-complex-fp64.cpp Depends device_math.h device_complex.h device.h clang)

set(devicelib-wrapper-cmath-spir64 ${obj_binary_dir}/libsycl-cmath-spir64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-spir64} Target spir64-unknown-unknown-sycldevice Source cmath_wrapper.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-spir64.bc Target spir64-unknown-unknown-sycldevice Source fallback-cmath.cpp Depends device_math.h device.h clang)

set(devicelib-wrapper-cmath-spir64_x86_64 ${obj_binary_dir}/libsycl-cmath-spir64_x86_64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-spir64_x86_64} Target spir64_x86_64-unknown-unknown-sycldevice Source cmath_wrapper.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-spir64_x86_64.bc Target spir64_x86_64-unknown-unknown-sycldevice Source fallback-cmath.cpp Depends device_math.h device.h clang)

set(devicelib-wrapper-cmath-spir64_gen ${obj_binary_dir}/libsycl-cmath-spir64_gen.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-spir64_gen} Target spir64_gen-unknown-unknown-sycldevice Source cmath_wrapper.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-spir64_gen.bc Target spir64_gen-unknown-unknown-sycldevice Source fallback-cmath.cpp Depends device_math.h device.h clang)

set(devicelib-wrapper-cmath-spir64_fpga ${obj_binary_dir}/libsycl-cmath-spir64_fpga.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-spir64_fpga} Target spir64_fpga-unknown-unknown-sycldevice Source cmath_wrapper.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-spir64_fpga.bc Target spir64_fpga-unknown-unknown-sycldevice Source fallback-cmath.cpp Depends device_math.h device.h clang)

set(devicelib-wrapper-cmath-fp64-spir64 ${obj_binary_dir}/libsycl-cmath-fp64-spir64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-fp64-spir64} Target spir64-unknown-unknown-sycldevice Source cmath_wrapper_fp64.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64.bc Target spir64-unknown-unknown-sycldevice Source fallback-cmath-fp64.cpp Depends device_math.h device.h clang)

set(devicelib-wrapper-cmath-fp64-spir64_x86_64 ${obj_binary_dir}/libsycl-cmath-fp64-spir64_x86_64.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-fp64-spir64_x86_64} Target spir64_x86_64-unknown-unknown-sycldevice Source cmath_wrapper_fp64.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_x86_64.bc Target spir64_x86_64-unknown-unknown-sycldevice Source fallback-cmath-fp64.cpp Depends device_math.h device.h clang)

set(devicelib-wrapper-cmath-fp64-spir64_gen ${obj_binary_dir}/libsycl-cmath-fp64-spir64_gen.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-fp64-spir64_gen} Target spir64_gen-unknown-unknown-sycldevice Source cmath_wrapper_fp64.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_gen.bc Target spir64_gen-unknown-unknown-sycldevice Source fallback-cmath-fp64.cpp Depends device_math.h device.h clang)

set(devicelib-wrapper-cmath-fp64-spir64_fpga ${obj_binary_dir}/libsycl-cmath-fp64-spir64_fpga.bc)
BuildSYCLDeviceLib(Path ${devicelib-wrapper-cmath-fp64-spir64_fpga} Target spir64_fpga-unknown-unknown-sycldevice Source cmath_wrapper_fp64.cpp Depends device_math.h device.h clang)
BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_fpga.bc Target spir64_fpga-unknown-unknown-sycldevice Source fallback-cmath-fp64.cpp Depends device_math.h device.h clang)

function(BuildSYCLFallbackDeviceLib)
  cmake_parse_arguments(DeviceLib "" "Path;Source" "Depends" ${ARGN})
  add_custom_command(OUTPUT ${DeviceLib_Path}
                     COMMAND ${clang} -S -fsycl-device-only -fno-sycl-use-bitcode
                             ${compile_opts}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${DeviceLib_Source}
                             -o ${DeviceLib_Path} -fno-sycl-device-lib=all
                     MAIN_DEPENDENCY ${DeviceLib_Source}
                     DEPENDS ${DeviceLib_Depends}
                     VERBATIM)
endfunction()

BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-cassert.spv Source fallback-cassert.cpp Depends wrapper.h device.h clang spirv_vars.h llvm-spirv)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-complex.spv Source fallback-complex.cpp Depends device_math.h device_complex.h device.h clang llvm-spirv)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv Source fallback-complex-fp64.cpp Depends device_math.h device_complex.h device.h clang llvm-spirv)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-cmath.spv Source fallback-cmath.cpp Depends device_math.h device.h clang llvm-spirv)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv Source fallback-cmath-fp64.cpp Depends device_math.h device.h clang llvm-spirv)

add_custom_target(libsycldevice-obj DEPENDS
  ${devicelib-wrapper-crt-spir64}
  ${devicelib-wrapper-crt-spir64_x86_64}
  ${devicelib-wrapper-crt-spir64_gen}
  ${devicelib-wrapper-crt-spir64_fpga}
  ${devicelib-wrapper-complex-spir64}
  ${devicelib-wrapper-complex-spir64_x86_64}
  ${devicelib-wrapper-complex-spir64_gen}
  ${devicelib-wrapper-complex-spir64_fpga}
  ${devicelib-wrapper-complex-fp64-spir64}
  ${devicelib-wrapper-complex-fp64-spir64_x86_64}
  ${devicelib-wrapper-complex-fp64-spir64_gen}
  ${devicelib-wrapper-complex-fp64-spir64_fpga}
  ${devicelib-wrapper-cmath-spir64}
  ${devicelib-wrapper-cmath-spir64_x86_64}
  ${devicelib-wrapper-cmath-spir64_gen}
  ${devicelib-wrapper-cmath-spir64_fpga}
  ${devicelib-wrapper-cmath-fp64-spir64}
  ${devicelib-wrapper-cmath-fp64-spir64_x86_64}
  ${devicelib-wrapper-cmath-fp64-spir64_gen}
  ${devicelib-wrapper-cmath-fp64-spir64_fpga}
)
add_custom_target(libsycldevice-spv DEPENDS
  ${spv_binary_dir}/libsycl-fallback-cassert.spv
  ${spv_binary_dir}/libsycl-fallback-complex.spv
  ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
  ${spv_binary_dir}/libsycl-fallback-cmath.spv
  ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
  )
add_custom_target(libsycldevice-fallback-obj DEPENDS
  ${obj_binary_dir}/libsycl-fallback-cassert-spir64.bc
  ${obj_binary_dir}/libsycl-fallback-cassert-spir64_x86_64.bc
  ${obj_binary_dir}/libsycl-fallback-cassert-spir64_gen.bc
  ${obj_binary_dir}/libsycl-fallback-cassert-spir64_fpga.bc
  ${obj_binary_dir}/libsycl-fallback-complex-spir64.bc
  ${obj_binary_dir}/libsycl-fallback-complex-spir64_x86_64.bc
  ${obj_binary_dir}/libsycl-fallback-complex-spir64_gen.bc
  ${obj_binary_dir}/libsycl-fallback-complex-spir64_fpga.bc
  ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64.bc
  ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_x86_64.bc
  ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_gen.bc
  ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_fpga.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-spir64.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-spir64_x86_64.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-spir64_gen.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-spir64_fpga.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_x86_64.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_gen.bc
  ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_fpga.bc
)
add_custom_target(libsycldevice DEPENDS
  libsycldevice-obj
  libsycldevice-fallback-obj
  libsycldevice-spv)

# Place device libraries near the libsycl.so library in an install
# directory as well
if (WIN32)
  set(install_dest_spv bin)
else()
  set(install_dest_spv lib${LLVM_LIBDIR_SUFFIX})
endif()

set(install_dest_lib lib${LLVM_LIBDIR_SUFFIX})

install(FILES ${devicelib-wrapper-crt-spir64}
              ${devicelib-wrapper-crt-spir64_x86_64}
              ${devicelib-wrapper-crt-spir64_gen}
              ${devicelib-wrapper-crt-spir64_fpga}
	      ${obj_binary_dir}/libsycl-fallback-cassert-spir64.bc
	      ${obj_binary_dir}/libsycl-fallback-cassert-spir64_x86_64.bc
	      ${obj_binary_dir}/libsycl-fallback-cassert-spir64_gen.bc
	      ${obj_binary_dir}/libsycl-fallback-cassert-spir64_fpga.bc
              ${devicelib-wrapper-complex-spir64}
              ${devicelib-wrapper-complex-spir64_x86_64}
              ${devicelib-wrapper-complex-spir64_gen}
              ${devicelib-wrapper-complex-spir64_fpga}
	      ${obj_binary_dir}/libsycl-fallback-complex-spir64.bc
	      ${obj_binary_dir}/libsycl-fallback-complex-spir64_x86_64.bc
	      ${obj_binary_dir}/libsycl-fallback-complex-spir64_gen.bc
	      ${obj_binary_dir}/libsycl-fallback-complex-spir64_fpga.bc
              ${devicelib-wrapper-complex-fp64-spir64}
              ${devicelib-wrapper-complex-fp64-spir64_x86_64}
              ${devicelib-wrapper-complex-fp64-spir64_gen}
              ${devicelib-wrapper-complex-fp64-spir64_fpga}
	      ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64.bc
	      ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_x86_64.bc
	      ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_gen.bc
	      ${obj_binary_dir}/libsycl-fallback-complex-fp64-spir64_fpga.bc
              ${devicelib-wrapper-cmath-spir64}
              ${devicelib-wrapper-cmath-spir64_x86_64}
              ${devicelib-wrapper-cmath-spir64_gen}
              ${devicelib-wrapper-cmath-spir64_fpga}
	      ${obj_binary_dir}/libsycl-fallback-cmath-spir64.bc
	      ${obj_binary_dir}/libsycl-fallback-cmath-spir64_x86_64.bc
	      ${obj_binary_dir}/libsycl-fallback-cmath-spir64_gen.bc
	      ${obj_binary_dir}/libsycl-fallback-cmath-spir64_fpga.bc
              ${devicelib-wrapper-cmath-fp64-spir64}
              ${devicelib-wrapper-cmath-fp64-spir64_x86_64}
              ${devicelib-wrapper-cmath-fp64-spir64_gen}
              ${devicelib-wrapper-cmath-fp64-spir64_fpga}
	      ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64.bc
	      ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_x86_64.bc
	      ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_gen.bc
	      ${obj_binary_dir}/libsycl-fallback-cmath-fp64-spir64_fpga.bc
        DESTINATION ${install_dest_lib}
        COMPONENT libsycldevice)

install(FILES ${spv_binary_dir}/libsycl-fallback-cassert.spv
              ${spv_binary_dir}/libsycl-fallback-complex.spv
              ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
              ${spv_binary_dir}/libsycl-fallback-cmath.spv
              ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
        DESTINATION ${install_dest_spv}
        COMPONENT libsycldevice)
