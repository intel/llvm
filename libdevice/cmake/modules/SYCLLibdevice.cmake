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
                     DEPENDS ${DeviceLib_Depends} clang device.h
                     VERBATIM)
endfunction()

set(SYCLWrapperLibs crt complex complex-fp64 cmath cmath-fp64)
set(SYCLFallbackLibs cassert complex complex-fp64 cmath cmath-fp64)
set(SYCLTargetTriples spir64 spir64_x86_64 spir64_gen spir64_fpga)

foreach(TTriple ${SYCLTargetTriples})
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-crt-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source ${libcrt_source} Depends devicelib_assert.h spirv_vars.h)
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cassert-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source fallback-cassert.cpp Depends devicelib_assert.h spirv_vars.h)
endforeach()

foreach(TTriple ${SYCLTargetTriples})
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-complex-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source complex_wrapper.cpp Depends device_complex.h)
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source fallback-complex.cpp Depends device_math.h device_complex.h)
endforeach()

foreach(TTriple ${SYCLTargetTriples})
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-complex-fp64-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source complex_wrapper_fp64.cpp Depends device_complex.h)
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-complex-fp64-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source fallback-complex-fp64.cpp Depends device_math.h device_complex.h)
endforeach()

foreach(TTriple ${SYCLTargetTriples})
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-cmath-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source cmath_wrapper.cpp Depends device_math.h)
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source fallback-cmath.cpp Depends device_math.h)
endforeach()

foreach(TTriple ${SYCLTargetTriples})
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-cmath-fp64-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source cmath_wrapper_fp64.cpp Depends device_math.h)
  BuildSYCLDeviceLib(Path ${obj_binary_dir}/libsycl-fallback-cmath-fp64-${TTriple}.bc Target ${TTriple}-unknown-unknown-sycldevice Source fallback-cmath-fp64.cpp Depends device_math.h)
endforeach()

function(BuildSYCLFallbackDeviceLib)
  cmake_parse_arguments(DeviceLib "" "Path;Source" "Depends" ${ARGN})
  add_custom_command(OUTPUT ${DeviceLib_Path}
                     COMMAND ${clang} -fsycl -fsycl-device-only -fno-sycl-use-bitcode
                             ${compile_opts}
                             ${CMAKE_CURRENT_SOURCE_DIR}/${DeviceLib_Source}
                             -o ${DeviceLib_Path} -fno-sycl-device-lib=all
                     MAIN_DEPENDENCY ${DeviceLib_Source}
                     DEPENDS ${DeviceLib_Depends} llvm-spirv clang device.h
                     VERBATIM)
endfunction()

BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-cassert.spv Source fallback-cassert.cpp Depends devicelib_assert.h spirv_vars.h)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-complex.spv Source fallback-complex.cpp Depends device_math.h device_complex.h)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv Source fallback-complex-fp64.cpp Depends device_math.h device_complex.h)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-cmath.spv Source fallback-cmath.cpp Depends device_math.h)
BuildSYCLFallbackDeviceLib(Path ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv Source fallback-cmath-fp64.cpp Depends device_math.h)

set(sycl-wrapper-lib-files "")
foreach(SYCLLib ${SYCLWrapperLibs})
  foreach(TTriple ${SYCLTargetTriples})
    list(APPEND sycl-wrapper-lib-files ${obj_binary_dir}/libsycl-${SYCLLib}-${TTriple}.bc)
  endforeach()
endforeach()

set(sycl-fallback-lib-files "")
foreach(SYCLLib ${SYCLFallbackLibs})
  foreach(TTriple ${SYCLTargetTriples})
    list(APPEND sycl-fallback-lib-files ${obj_binary_dir}/libsycl-fallback-${SYCLLib}-${TTriple}.bc)
  endforeach()
endforeach()

add_custom_target(libsycldevice-obj DEPENDS ${sycl-wrapper-lib-files})
add_custom_target(libsycldevice-fallback-obj DEPENDS ${sycl-fallback-lib-files})
add_custom_target(libsycldevice-spv DEPENDS
  ${spv_binary_dir}/libsycl-fallback-cassert.spv
  ${spv_binary_dir}/libsycl-fallback-complex.spv
  ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
  ${spv_binary_dir}/libsycl-fallback-cmath.spv
  ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
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

install(FILES ${sycl-wrapper-lib-files} ${sycl-fallback-lib-files}
        DESTINATION ${install_dest_lib}
        COMPONENT libsycldevice)

install(FILES ${spv_binary_dir}/libsycl-fallback-cassert.spv
              ${spv_binary_dir}/libsycl-fallback-complex.spv
              ${spv_binary_dir}/libsycl-fallback-complex-fp64.spv
              ${spv_binary_dir}/libsycl-fallback-cmath.spv
              ${spv_binary_dir}/libsycl-fallback-cmath-fp64.spv
        DESTINATION ${install_dest_spv}
        COMPONENT libsycldevice)
