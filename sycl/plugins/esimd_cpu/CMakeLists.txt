
# PI Esimd CPU  library
# Create Shared library for libpi_esimd_cpu.so.

include(ExternalProject)

include_directories("${sycl_inc_dir}")
# FIXME/TODO: 'pi.h' is included in 'pi_esimd_cpu.cpp', and CL_*_INTEL
# and CL_*_KHR definitions in 'pi.h' are from
# ${OPENCL_INCLUDE}. Remove build dependency on OpenCL
include_directories(${OpenCL_INCLUDE_DIR})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install)

if (MSVC)
  set(LIBCM ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/lib/libcm${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(LIBIGFXCMRT_EMU ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/lib/igfxcmrt64_emu${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(LIBCM ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/lib/libcm${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(LIBIGFXCMRT_EMU ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/lib/libigfxcmrt_emu${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if (DEFINED CM_LOCAL_SOURCE_DIR)
    # Using local CM directory for online building without downloading
  if (MSVC)
    ExternalProject_Add(cm-emu
      DOWNLOAD_COMMAND  ""
      SOURCE_DIR        ${CM_LOCAL_SOURCE_DIR}
      BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build
      INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
      CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
    )
  else()
    ExternalProject_Add(cm-emu
      DOWNLOAD_COMMAND  ""
      SOURCE_DIR        ${CM_LOCAL_SOURCE_DIR}
      BINARY_DIR        ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_build
      INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
      CMAKE_ARGS        -DLIBVA_INSTALL_PATH=/usr
                        -D__SYCL_EXPLICIT_SIMD_PLUGIN__=true
                        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
    )
  endif()
else ()
  # Downloading pre-built CM Package
  if (NOT DEFINED CM_PACKAGE_URL)
    if (MSVC)
      message(FATAL_ERROR "ESIMD_CPU is not supported under Windows environment yet")
    else()
      # FIXME/TODO : Use 'generic' package instead of 'u18.04'
      set (CM_PACKAGE_URL "https://github.com/intel/cm-cpu-emulation/releases/download/v2021-07-21/intel-cmemu-1.0.1.u18.04-release.x86_64.tar.xz")
    endif()
  endif()
  file (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install)
  ExternalProject_Add(cm-emu
    URL               ${CM_PACKAGE_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    UPDATE_COMMAND    ""
    INSTALL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install
    INSTALL_COMMAND   ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/cm-emu-prefix/src/cm-emu/ <INSTALL_DIR>
    BUILD_BYPRODUCTS  ${LIBCM} ${LIBIGFXCMRT_EMU}
  )
endif ()
ExternalProject_Add_Step(cm-emu llvminstall
  COMMAND ${CMAKE_COMMAND} -E make_directory ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps && ${CMAKE_COMMAND} -E copy_directory <INSTALL_DIR>/ ${LLVM_BINARY_DIR}/pi_esimd_cpu_deps
  COMMENT "Installing cm-emu into the LLVM binary directory"
  DEPENDEES install
)

include_directories(${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/include/igfxcmrt_emu)
include_directories(${LLVM_BINARY_DIR}/pi_esimd_cpu_deps/include/libcm/cm)

# Compilation flag to exclude lines in header files imported from CM
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__SYCL_EXPLICIT_SIMD_PLUGIN__")

set(CMAKE_CXX_STANDARD 17)

# Compilation option modification to prevent build termination caused by
# warnings from CM-imported files
if (MSVC)
string(REPLACE "/W4" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
string(REPLACE "-pedantic" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_library(pi_esimd_cpu SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "pi_esimd_cpu.cpp"
)

if (MSVC)
  # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
  # which are individually tagged for all pi* symbols in pi.h
  target_compile_definitions(pi_esimd_cpu PRIVATE __SYCL_BUILD_SYCL_DLL)
else()
  # we set the visibility of all symbols 'hidden' by default.
  # In pi.h file, we set exported symbols with visibility==default individually
  target_compile_options(pi_esimd_cpu PUBLIC -fvisibility=hidden)

  # This script file is used to allow exporting pi* symbols only.
  # All other symbols are regarded as local (hidden)
  set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt")

  # Filter symbols based on the scope defined in the script file,
  # and export pi* function symbols in the library.
  target_link_libraries( pi_esimd_cpu
    PRIVATE "-Wl,--version-script=${linker_script}"
  )
endif()

add_dependencies(pi_esimd_cpu OpenCL-Headers)
add_dependencies(pi_esimd_cpu cm-emu)
add_dependencies(sycl-toolchain pi_esimd_cpu)

target_link_libraries(pi_esimd_cpu PRIVATE sycl ${LIBCM} ${LIBIGFXCMRT_EMU})
set_target_properties(pi_esimd_cpu PROPERTIES LINKER_LANGUAGE CXX)

add_common_options(pi_esimd_cpu)

install(TARGETS pi_esimd_cpu
        LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_esimd_cpu
        RUNTIME DESTINATION "bin" COMPONENT pi_esimd_cpu)

# Copy CM Header files to $(INSTALL)/include/sycl/CL/
install(DIRECTORY    ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/include/libcm/cm/
        DESTINATION  ${SYCL_INCLUDE_DIR}/CL
        COMPONENT    libcmrt-headers
        FILES_MATCHING PATTERN "*.h"
)

# Copy '.so' files to '$(INSTALL)/lib'
if (MSVC)
  install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/lib/
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
          COMPONENT   libcmrt-libs
          FILES_MATCHING PATTERN "*.lib"
  )
  install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/bin/
          DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
          COMPONENT   libcmrt-dlls
          FILES_MATCHING PATTERN "*.dll"
  )
else()
  install(DIRECTORY   ${CMAKE_CURRENT_BINARY_DIR}/cm-emu_install/lib/
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
          COMPONENT   libcmrt-sos
          FILES_MATCHING PATTERN "*.so"
  )
endif()
