# INTEL_COLLAB
##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Plugin for OpenCL device
#
##===----------------------------------------------------------------------===##

if(LIBOMPTARGET_DEP_OPENCL_FOUND)
    if((CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)$" OR
        CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64") AND
       (CMAKE_SYSTEM_NAME MATCHES "Linux" OR
        CMAKE_SYSTEM_NAME MATCHES "Windows"))

        libomptarget_say("Building OpenCL offloading plugin.")

        # Define the suffix for the runtime messaging dumps.
        add_definitions(-DTARGET_NAME=OPENCL)
        add_definitions(-DCL_TARGET_OPENCL_VERSION=210)

        if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
            add_definitions(-DSPIR_ERROR_REPORT)
            add_definitions(-DOMPTARGET_OPENCL_DEBUG)
        endif()

        include_directories(${LIBOMPTARGET_DEP_OPENCL_INCLUDE_DIRS})
        add_library(omptarget.rtl.opencl SHARED src/rtl.cpp src/ompt-opencl.cpp)
        if(NOT LIBOMPTARGET_DEP_LIBELF_FOUND OR LIBOMPTARGET_FORCE_LLVM_ELF)
          # Statically link C run-time library on Windows.
          # Otherwise, we cannot use LLVMObject and LLVMSupport.
          set_msvc_crt_flags(omptarget.rtl.opencl)
        endif()

        if (NOT OPENMP_STANDALONE_BUILD)
          # We have to disable EH for Windows compilation.
          # For standalone OpenMP build, we need to come up
          # with our own EH flags management.
          llvm_update_compile_flags(omptarget.rtl.opencl)
        endif()

        # Install plugin under the lib destination folder.
        # DLL part (for DLL platforms) is installed into bin.
        # Do not install interface library for DLL platforms,
        # since the plugin is only supposed to be used by libomptarget.
        install(TARGETS omptarget.rtl.opencl
                LIBRARY COMPONENT llvm-opencl-ompplugin
                DESTINATION lib${LIBOMPTARGET_LIBDIR_SUFFIX}
                RUNTIME COMPONENT llvm-opencl-ompplugin
                DESTINATION bin)

        # OpenMP runtime library must be linked in.
        target_link_libraries(omptarget.rtl.opencl PRIVATE omp)

        target_link_libraries(omptarget.rtl.opencl PRIVATE
          elf_common
          ${LIBOMPTARGET_DEP_OPENCL_LIBRARIES})

        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
          target_link_libraries(omptarget.rtl.opencl PRIVATE
            dl
            ${OPENMP_PTHREAD_LIB}
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")
        elseif (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
          message(FATAL_ERROR "Missing platfrom support.")
        endif()

        # Report to the parent scope that we are building a plugin for OpenCL.
        set(LIBOMPTARGET_SYSTEM_TARGETS
          "${LIBOMPTARGET_SYSTEM_TARGETS} spir64-unknown-unknown" PARENT_SCOPE)
    else()
        libomptarget_say("Not building OpenCL offloading plugin: \
                          only support OpenCL in x86_64 Linux/Windows hosts.")
    endif()
else()
    libomptarget_say("Not building OpenCL offloading plugin: \
                      OpenCL not found in system.")
endif()

# Enable new IN_LIST operator.
cmake_policy(SET CMP0057 NEW)
if (libdevice IN_LIST LLVM_ENABLE_PROJECTS)
  add_dependencies(omptarget.rtl.opencl libompdevice)
endif()
# end INTEL_COLLAB
