set(obj_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(obj-new-offload_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if (MSVC)
  set(obj-suffix obj)
  set(obj-new-offload-suffix new.obj)
  set(spv_binary_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  set(install_dest_spv bin)
  set(devicelib_host_static_obj sycl-devicelib-host.lib)
  set(devicelib_host_static_obj-new-offload sycl-devicelib-host.new.lib)
else()
  set(obj-suffix o)
  set(obj-new-offload-suffix new.o)
  set(spv_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(install_dest_spv lib${LLVM_LIBDIR_SUFFIX})
  set(devicelib_host_static_obj libsycl-devicelib-host.a)
  set(devicelib_host_static_obj-new-offload libsycl-devicelib-host.new.a)
endif()
set(spv-suffix spv)
set(bc-suffix bc)
set(bc_binary_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
set(install_dest_obj lib${LLVM_LIBDIR_SUFFIX})
set(install_dest_obj-new-offload lib${LLVM_LIBDIR_SUFFIX})
set(install_dest_bc lib${LLVM_LIBDIR_SUFFIX})

string(CONCAT sycl_targets_opt
  "-fsycl-targets="
  "spir64_x86_64-unknown-unknown,"
  "spir64_gen-unknown-unknown,"
  "spir64_fpga-unknown-unknown,"
  "spir64-unknown-unknown,"
  "spirv64-unknown-unknown")

string(CONCAT sycl_pvc_target_opt
  "-fsycl-targets="
  "intel_gpu_pvc")

string(CONCAT sycl_cpu_target_opt
  "-fsycl-targets="
  "spir64_x86_64-unknown-unknown")

string(CONCAT sycl_dg2_target_opt
  "-fsycl-targets="
  "spir64_gen-unknown-unknown")

set(compile_opts
  # suppress an error about SYCL_EXTERNAL being used for
  # a function with a raw pointer parameter.
  -Wno-sycl-strict
  # Disable warnings for the host compilation, where
  # we declare all functions as 'static'.
  -Wno-undefined-internal
  -sycl-std=2020
  --target=${LLVM_HOST_TRIPLE}
  )

set(SYCL_LIBDEVICE_GCC_TOOLCHAIN "" CACHE PATH "Path to GCC installation")

if (NOT SYCL_LIBDEVICE_GCC_TOOLCHAIN STREQUAL "")
  list(APPEND compile_opts "--gcc-toolchain=${SYCL_LIBDEVICE_GCC_TOOLCHAIN}")
endif()

if (WIN32)
  list(APPEND compile_opts -D_ALLOW_RUNTIME_LIBRARY_MISMATCH)
  list(APPEND compile_opts -D_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
endif()

add_custom_target(libsycldevice)

set(filetypes obj obj-new-offload spv bc)

foreach(filetype IN LISTS filetypes)
  add_custom_target(libsycldevice-${filetype})
  add_dependencies(libsycldevice libsycldevice-${filetype})
endforeach()

# For NVPTX and AMDGCN each device libary is compiled into a single bitcode
# file and all files created this way are linked into one large bitcode
# library.
# Additional compilation options are needed for compiling each device library.
set(devicelib_arch)
if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND devicelib_arch nvptx64-nvidia-cuda)
  set(compile_opts_nvptx64-nvidia-cuda "-fsycl-targets=nvptx64-nvidia-cuda"
  "-Xsycl-target-backend" "--cuda-gpu-arch=sm_50" "-nocudalib")
  set(opt_flags_nvptx64-nvidia-cuda "-O3" "--nvvm-reflect-enable=false")
endif()
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND devicelib_arch amdgcn-amd-amdhsa)
  set(compile_opts_amdgcn-amd-amdhsa "-nogpulib" "-fsycl-targets=amdgcn-amd-amdhsa"
  "-Xsycl-target-backend" "--offload-arch=gfx940")
  set(opt_flags_amdgcn-amd-amdhsa "-O3" "--amdgpu-oclc-reflect-enable=false")
endif()


set(spv_device_compile_opts -fsycl-device-only -fsycl-device-obj=spirv)
set(bc_device_compile_opts -fsycl-device-only -fsycl-device-obj=llvmir)
set(obj-new-offload_device_compile_opts -fsycl -c --offload-new-driver
  -foffload-lto=thin ${sycl_targets_opt})
set(obj_device_compile_opts -fsycl -c ${sycl_targets_opt})

# Compiles and installs a single device library.
#
# Arguments:
# * FILETYPE <string>
#     Specifies the output file type of the compilation and its repsective
#     installation directory.
#     Adds a new target that the libsycldevice-FILETYPE target will depend on.
# * SRC <string> ...
#    Source code files needed for the compilation.
# * EXTRA_OPTS <string> ...
#     List of extra compiler options to use.
#     Note that the ones specified by the compile_opts var are always used.
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
#
# Depends on the clang target for compiling.
function(compile_lib filename)
  cmake_parse_arguments(ARG
    ""
    "FILETYPE"
    "SRC;EXTRA_OPTS;DEPENDENCIES"
    ${ARGN})
    set(compile_opt_list ${compile_opts}
                         ${${ARG_FILETYPE}_device_compile_opts}
                         ${ARG_EXTRA_OPTS})
    compile_lib_ext(${filename}
      FILETYPE ${ARG_FILETYPE}
      SRC ${ARG_SRC}
      DEPENDENCIES ${ARG_DEPENDENCIES}
      OPTS ${compile_opt_list})
endfunction()

function(compile_lib_ext filename)
  cmake_parse_arguments(ARG
    ""
    "FILETYPE"
    "SRC;OPTS;DEPENDENCIES"
    ${ARGN})

  set(devicelib-file
    ${${ARG_FILETYPE}_binary_dir}/${filename}.${${ARG_FILETYPE}-suffix})

  add_custom_command(
    OUTPUT ${devicelib-file}
    COMMAND ${clang_exe} -I ${PROJECT_BINARY_DIR}/include
            ${ARG_OPTS}
            ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SRC} -o ${devicelib-file}
    MAIN_DEPENDENCY ${ARG_SRC}
    DEPENDS ${ARG_DEPENDENCIES}
    VERBATIM
  )
  set(devicelib-${ARG_FILETYPE}-target ${filename}-${ARG_FILETYPE})
  add_custom_target(${devicelib-${ARG_FILETYPE}-target}
    DEPENDS ${devicelib-file})
  add_dependencies(libsycldevice-${ARG_FILETYPE}
    ${devicelib-${ARG_FILETYPE}-target})

  install( FILES ${devicelib-file}
           DESTINATION ${install_dest_${ARG_FILETYPE}}
           COMPONENT libsycldevice)
endfunction()

# Appends a list to a global property.
#
# Arguments:
# * PROPERTY_NAME <string>
#     The name of the property to append to.
function(append_to_property list)
  cmake_parse_arguments(ARG
    ""
    "PROPERTY_NAME"
    ""
    ${ARGN})
  get_property(new_property GLOBAL PROPERTY ${ARG_PROPERTY_NAME})
  list(APPEND new_property ${list})
  set_property(GLOBAL PROPERTY ${ARG_PROPERTY_NAME} ${new_property})
endfunction()

# Creates device libaries for all filetypes.
# Adds bitcode library files additionally for each devicelib_arch target and
# adds the created file to an arch specific global property.
#
# Arguments:
# * SRC <string> ...
#    Source code files needed for the compilation.
# * EXTRA_OPTS <string> ...
#     List of extra compiler options to use.
#     Note that the ones specified by the compile_opts var are always used.
# * DEPENDENCIES <string> ...
#     List of extra dependencies to inject
#
# Depends on the clang target for compiling.
function(add_devicelibs filename)
  cmake_parse_arguments(ARG
    ""
    ""
    "SRC;EXTRA_OPTS;DEPENDENCIES;SKIP_ARCHS"
    ${ARGN})

  foreach(filetype IN LISTS filetypes)
    compile_lib(${filename}
      FILETYPE ${filetype}
      SRC ${ARG_SRC}
      DEPENDENCIES ${ARG_DEPENDENCIES}
      EXTRA_OPTS ${ARG_EXTRA_OPTS} ${${filetype}_device_compile_opts})
  endforeach()

  foreach(arch IN LISTS devicelib_arch)
    if(arch IN_LIST ARG_SKIP_ARCHS)
      continue()
    endif()
    compile_lib(${filename}-${arch}
      FILETYPE bc
      SRC ${ARG_SRC}
      DEPENDENCIES ${ARG_DEPENDENCIES}
      EXTRA_OPTS ${ARG_EXTRA_OPTS} ${bc_device_compile_opts}
                 ${compile_opts_${arch}})

    append_to_property(${bc_binary_dir}/${filename}-${arch}.bc
      PROPERTY_NAME BC_DEVICE_LIBS_${arch})
  endforeach()
endfunction()

# For native builds, sycl-compiler will already include everything we need.
# For cross builds, we also need native versions of the tools.
set(sycl-compiler_deps
  sycl-compiler ${clang_target} ${append-file_target}
  ${clang-offload-bundler_target} ${clang-offload-packager_target}
  ${file-table-tform_target} ${llvm-foreach_target} ${llvm-spirv_target}
  ${sycl-post-link_target})
set(crt_obj_deps wrapper.h device.h spirv_vars.h ${sycl-compiler_deps})
set(complex_obj_deps device_complex.h device.h ${sycl-compiler_deps})
set(cmath_obj_deps device_math.h device.h ${sycl-compiler_deps})
set(imf_obj_deps device_imf.hpp imf_half.hpp imf_bf16.hpp imf_rounding_op.hpp imf_impl_utils.hpp device.h ${sycl-compiler_deps})
set(itt_obj_deps device_itt.h spirv_vars.h device.h ${sycl-compiler_deps})
set(bfloat16_obj_deps sycl-headers ${sycl-compiler_deps})
if (NOT MSVC AND UR_SANITIZER_INCLUDE_DIR)
  set(asan_obj_deps
    device.h atomic.hpp spirv_vars.h
    ${UR_SANITIZER_INCLUDE_DIR}/asan/asan_libdevice.hpp
    include/asan_rtl.hpp
    include/spir_global_var.hpp
    ${sycl-compiler_deps})

  set(sanitizer_generic_compile_opts ${compile_opts}
                            -fno-sycl-instrument-device-code
                            -I${UR_SANITIZER_INCLUDE_DIR}
                            -I${CMAKE_CURRENT_SOURCE_DIR})

  set(asan_pvc_compile_opts_obj -fsycl -c
                                ${sanitizer_generic_compile_opts}
                                ${sycl_pvc_target_opt}
                                -D__LIBDEVICE_PVC__)

  set(asan_cpu_compile_opts_obj -fsycl -c
                                ${sanitizer_generic_compile_opts}
                                ${sycl_cpu_target_opt}
                                -D__LIBDEVICE_CPU__)

  set(asan_dg2_compile_opts_obj -fsycl -c
                                ${sanitizer_generic_compile_opts}
                                ${sycl_dg2_target_opt}
                                -D__LIBDEVICE_DG2__)

  set(asan_pvc_compile_opts_bc  ${bc_device_compile_opts}
                                ${sanitizer_generic_compile_opts}
                                -D__LIBDEVICE_PVC__)

  set(asan_cpu_compile_opts_bc  ${bc_device_compile_opts}
                                ${sanitizer_generic_compile_opts}
                                -D__LIBDEVICE_CPU__)

  set(asan_dg2_compile_opts_bc  ${bc_device_compile_opts}
                                ${sanitizer_generic_compile_opts}
                                -D__LIBDEVICE_DG2__)

  set(asan_pvc_compile_opts_obj-new-offload -fsycl -c --offload-new-driver
                                            -foffload-lto=thin
                                            ${sanitizer_generic_compile_opts}
                                            ${sycl_pvc_target_opt}
                                            -D__LIBDEVICE_PVC__)

  set(asan_cpu_compile_opts_obj-new-offload -fsycl -c --offload-new-driver
                                            -foffload-lto=thin
                                            ${sanitizer_generic_compile_opts}
                                            ${sycl_cpu_target_opt}
                                            -D__LIBDEVICE_CPU__)

  set(asan_dg2_compile_opts_obj-new-offload -fsycl -c --offload-new-driver
                                            -foffload-lto=thin
                                            ${sanitizer_generic_compile_opts}
                                            ${sycl_dg2_target_opt}
                                            -D__LIBDEVICE_DG2__)
endif()

if("native_cpu" IN_LIST SYCL_ENABLE_BACKENDS)
  if (NOT DEFINED NATIVE_CPU_DIR)
    message( FATAL_ERROR "Undefined UR variable NATIVE_CPU_DIR. The name may have changed." )
  endif()
  # Include NativeCPU UR adapter path to enable finding header file with state struct.
  # libsycl-nativecpu_utils is only needed as BC file by NativeCPU.
  # Todo: add versions for other targets (for cross-compilation)
  compile_lib(libsycl-nativecpu_utils
    FILETYPE bc
    SRC nativecpu_utils.cpp
    DEPENDENCIES ${itt_obj_deps}
    EXTRA_OPTS -I ${NATIVE_CPU_DIR} -fsycl-targets=native_cpu -fsycl-device-only
               -fsycl-device-obj=llvmir)
endif()

# Add all device libraries for each filetype except for the Intel math function
# ones.
add_devicelibs(libsycl-itt-stubs
  SRC itt_stubs.cpp
  DEPENDENCIES ${itt_obj_deps})
add_devicelibs(libsycl-itt-compiler-wrappers
  SRC itt_compiler_wrappers.cpp
  DEPENDENCIES ${itt_obj_deps})
add_devicelibs(libsycl-itt-user-wrappers
  SRC itt_user_wrappers.cpp
  DEPENDENCIES ${itt_obj_deps})

add_devicelibs(libsycl-crt
  SRC crt_wrapper.cpp
  DEPENDENCIES ${crt_obj_deps})
add_devicelibs(libsycl-complex
  SRC complex_wrapper.cpp
  DEPENDENCIES ${complex_obj_deps})
add_devicelibs(libsycl-complex-fp64
  SRC complex_wrapper_fp64.cpp
  DEPENDENCIES ${complex_obj_deps} )
add_devicelibs(libsycl-cmath
  SRC cmath_wrapper.cpp
  DEPENDENCIES ${cmath_obj_deps})
add_devicelibs(libsycl-cmath-fp64
  SRC cmath_wrapper_fp64.cpp
  DEPENDENCIES ${cmath_obj_deps} )
add_devicelibs(libsycl-imf
  SRC imf_wrapper.cpp
  DEPENDENCIES ${imf_obj_deps})
add_devicelibs(libsycl-imf-fp64
  SRC imf_wrapper_fp64.cpp
  DEPENDENCIES ${imf_obj_deps})
add_devicelibs(libsycl-imf-bf16
  SRC imf_wrapper_bf16.cpp
  DEPENDENCIES ${imf_obj_deps})
add_devicelibs(libsycl-bfloat16
  SRC bfloat16_wrapper.cpp
  DEPENDENCIES ${cmath_obj_deps})
if(MSVC)
  add_devicelibs(libsycl-msvc-math
    SRC msvc_math.cpp
    DEPENDENCIES ${cmath_obj_deps})
else()
  if(UR_SANITIZER_INCLUDE_DIR)
    # asan jit
    add_devicelibs(libsycl-asan
      SRC sanitizer/asan_rtl.cpp
      DEPENDENCIES ${asan_obj_deps}
      SKIP_ARCHS nvptx64-nvidia-cuda
                 amdgcn-amd-amdhsa
      EXTRA_OPTS -fno-sycl-instrument-device-code
                 -I${UR_SANITIZER_INCLUDE_DIR}
                 -I${CMAKE_CURRENT_SOURCE_DIR})

    # asan aot
    set(asan_filetypes obj obj-new-offload bc)
    set(asan_devicetypes pvc cpu dg2)

    foreach(asan_ft IN LISTS asan_filetypes)
      foreach(asan_device IN LISTS asan_devicetypes)
        compile_lib_ext(libsycl-asan-${asan_device}
                        SRC sanitizer/asan_rtl.cpp
                        FILETYPE ${asan_ft}
                        DEPENDENCIES ${asan_obj_deps}
                        OPTS ${asan_${asan_device}_compile_opts_${asan_ft}})
      endforeach()
    endforeach()
  endif()
endif()

add_devicelibs(libsycl-fallback-cassert
  SRC fallback-cassert.cpp
  DEPENDENCIES ${crt_obj_deps}
  EXTRA_OPTS -fno-sycl-instrument-device-code)
add_devicelibs(libsycl-fallback-cstring
  SRC fallback-cstring.cpp
  DEPENDENCIES ${crt_obj_deps})
add_devicelibs(libsycl-fallback-complex
  SRC fallback-complex.cpp
  DEPENDENCIES ${complex_obj_deps})
add_devicelibs(libsycl-fallback-complex-fp64
  SRC fallback-complex-fp64.cpp
  DEPENDENCIES ${complex_obj_deps})
add_devicelibs(libsycl-fallback-cmath
  SRC fallback-cmath.cpp
  DEPENDENCIES ${cmath_obj_deps})
add_devicelibs(libsycl-fallback-cmath-fp64
  SRC fallback-cmath-fp64.cpp
  DEPENDENCIES ${cmath_obj_deps})
add_devicelibs(libsycl-fallback-bfloat16
  SRC fallback-bfloat16.cpp
  DEPENDENCIES ${bfloat16_obj_deps})
add_devicelibs(libsycl-native-bfloat16
  SRC bfloat16_wrapper.cpp
  DEPENDENCIES ${bfloat16_obj_deps})

# Create dependency and source lists for Intel math function libraries.
file(MAKE_DIRECTORY ${obj_binary_dir}/libdevice)
set(imf_fallback_src_dir ${obj_binary_dir}/libdevice)
set(imf_src_dir ${CMAKE_CURRENT_SOURCE_DIR})
set(imf_fallback_fp32_deps device.h device_imf.hpp imf_half.hpp imf_rounding_op.hpp imf_impl_utils.hpp
                           imf_utils/integer_misc.cpp
                           imf_utils/float_convert.cpp
                           imf_utils/half_convert.cpp
                           imf_utils/simd_emulate.cpp
                           imf_utils/fp32_round.cpp
                           imf/imf_inline_fp32.cpp
                           imf/imf_fp32_dl.cpp)
set(imf_fallback_fp64_deps device.h device_imf.hpp imf_half.hpp imf_rounding_op.hpp imf_impl_utils.hpp
                           imf_utils/double_convert.cpp imf_utils/fp64_round.cpp
                           imf/imf_inline_fp64.cpp
                           imf/imf_fp64_dl.cpp)
set(imf_fallback_bf16_deps device.h device_imf.hpp imf_bf16.hpp
                           imf_utils/bfloat16_convert.cpp
                           imf/imf_inline_bf16.cpp)

set(imf_fp32_fallback_src ${imf_fallback_src_dir}/imf_fp32_fallback.cpp)
set(imf_fp64_fallback_src ${imf_fallback_src_dir}/imf_fp64_fallback.cpp)
set(imf_bf16_fallback_src ${imf_fallback_src_dir}/imf_bf16_fallback.cpp)

set(imf_host_cxx_flags -c
  --target=${LLVM_HOST_TRIPLE}
  -D__LIBDEVICE_HOST_IMPL__
)

macro(mangle_name str output)
  string(STRIP "${str}" strippedStr)
  string(REGEX REPLACE "^/" "" strippedStr "${strippedStr}")
  string(REGEX REPLACE "^-+" "" strippedStr "${strippedStr}")
  string(REGEX REPLACE "-+$" "" strippedStr "${strippedStr}")
  string(REPLACE "-" "_" strippedStr "${strippedStr}")
  string(REPLACE "=" "_EQ_" strippedStr "${strippedStr}")
  string(REPLACE "+" "X" strippedStr "${strippedStr}")
  string(TOUPPER "${strippedStr}" ${output})
endmacro()

# Add a list of flags to 'imf_host_cxx_flags'.
macro(add_imf_host_cxx_flags_compile_flags)
  foreach(f ${ARGN})
    list(APPEND imf_host_cxx_flags ${f})
  endforeach()
endmacro()

# If 'condition' is true then add the specified list of flags to
# 'imf_host_cxx_flags'
macro(add_imf_host_cxx_flags_compile_flags_if condition)
  if (${condition})
    add_imf_host_cxx_flags_compile_flags(${ARGN})
  endif()
endmacro()

# For each specified flag, add that flag to 'imf_host_cxx_flags' if the
# flag is supported by the C++ compiler.
macro(add_imf_host_cxx_flags_compile_flags_if_supported)
  foreach(flag ${ARGN})
      mangle_name("${flag}" flagname)
      check_cxx_compiler_flag("${flag}" "CXX_SUPPORTS_${flagname}_FLAG")
      add_imf_host_cxx_flags_compile_flags_if(CXX_SUPPORTS_${flagname}_FLAG ${flag})
  endforeach()
endmacro()


if (NOT WIN32)
  list(APPEND imf_host_cxx_flags -fPIC)
  add_imf_host_cxx_flags_compile_flags_if_supported("-fcf-protection=full")
endif()

set(obj-new-offload_host_compile_opts ${imf_host_cxx_flags} --offload-new-driver
  -foffload-lto=thin)
set(obj_host_compile_opts ${imf_host_cxx_flags})

foreach(datatype IN ITEMS fp32 fp64 bf16)
  string(TOUPPER ${datatype} upper_datatype)

  add_custom_command(
    OUTPUT ${imf_${datatype}_fallback_src}
    COMMAND ${CMAKE_COMMAND}
            -D SRC_DIR=${imf_src_dir}
            -D DEST_DIR=${imf_fallback_src_dir}
            -D IMF_TARGET=${upper_datatype}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/ImfSrcConcate.cmake
    DEPENDS ${imf_fallback_${datatype}_deps})

  add_custom_target(get_imf_fallback_${datatype}
     DEPENDS ${imf_${datatype}_fallback_src})
endforeach()

# Adds Intel math functions libraries.
#
# Arguments:
# * SRC <string> ...
#    Source code files needed for the compilation.
# * DIR <string>
#    The directory where the output file should be located in.
# * FTYPE <string>
#    Filetype of the output library file (e.g. 'bc').
# * DTYPE <string>
#   The datatype of the library, which determines the input source
#   and dependencies of the compilation command.
# * TGT_NAME <string>
#   Name of the new target that depends on the compilation of the library.
# * EXTRA_OPTS <string> ...
#     List of extra compiler options to use.
#     Note that the ones specified by the compile_opts var are always used.
#
# Depends on the clang target for compiling.
function(add_lib_imf name)
  cmake_parse_arguments(ARG
    ""
    "DIR;FTYPE;DTYPE;TGT_NAME"
    "EXTRA_OPTS"
    ${ARGN})

  add_custom_command(
    OUTPUT ${ARG_DIR}/${name}.${${ARG_FTYPE}-suffix}
    COMMAND ${clang_exe} ${compile_opts} ${ARG_EXTRA_OPTS}
            -I ${CMAKE_CURRENT_SOURCE_DIR}/imf
            ${imf_${ARG_DTYPE}_fallback_src}
            -o
            ${ARG_DIR}/${name}.${${ARG_FTYPE}-suffix}
            DEPENDS ${imf_fallback_${ARG_DTYPE}_deps}
            get_imf_fallback_${ARG_DTYPE} ${sycl-compiler_deps}
    VERBATIM)

  add_custom_target(${ARG_TGT_NAME}
    DEPENDS ${ARG_DIR}/${name}.${${ARG_FTYPE}-suffix})

  add_dependencies(libsycldevice-${ARG_FTYPE} ${ARG_TGT_NAME})
endfunction()

# Add device fallback imf libraries for the SPIRV targets and all filetypes.
foreach(dtype IN ITEMS bf16 fp32 fp64)
  foreach(ftype IN LISTS filetypes)
    set(libsycl_name libsycl-fallback-imf)
    if (NOT (dtype STREQUAL "fp32"))
      set(libsycl_name libsycl-fallback-imf-${dtype})
    endif()
    set(tgt_name imf_fallback_${dtype}_${ftype})

    add_lib_imf(${libsycl_name}
      DIR ${${ftype}_binary_dir}
      FTYPE ${ftype}
      DTYPE ${dtype}
      EXTRA_OPTS ${${ftype}_device_compile_opts}
      TGT_NAME ${tgt_name})
  endforeach()
endforeach()

# Add device fallback imf libraries for the NVPTX and AMD targets.
# The output files are bitcode.
foreach(arch IN LISTS devicelib_arch)
  foreach(dtype IN ITEMS bf16 fp32 fp64)
    set(tgt_name imf_fallback_${dtype}_bc_${arch})

    add_lib_imf(libsycl-fallback-imf-${arch}-${dtype}
      ARCH ${arch}
      DIR ${bc_binary_dir}
      FTYPE bc
      DTYPE ${dtype}
      EXTRA_OPTS ${bc_device_compile_opts} ${compile_opts_${arch}}
      TGT_NAME ${tgt_name})

    append_to_property(
      ${bc_binary_dir}/libsycl-fallback-imf-${arch}-${dtype}.${bc-suffix}
      PROPERTY_NAME ${arch})
  endforeach()
endforeach()

# Create one large bitcode file for the NVPTX and AMD targets.
# Use all the files collected in the respective global properties.
foreach(arch IN LISTS devicelib_arch)
  get_property(BC_DEVICE_LIBS_${arch} GLOBAL PROPERTY BC_DEVICE_LIBS_${arch})
  # Link the bitcode files together.
  link_bc(TARGET device_lib_device_${arch}
          RSP_DIR ${CMAKE_CURRENT_BINARY_DIR}
          INPUTS ${BC_DEVICE_LIBS_${arch}})
  set( builtins_link_lib_${arch}
    $<TARGET_PROPERTY:device_lib_device_${arch},TARGET_FILE>)
  add_dependencies(libsycldevice-bc device_lib_device_${arch})
  set( builtins_opt_lib_tgt_${arch} builtins_${arch}.opt)

  # Run the optimizer on the resulting bitcode file and call prepare_builtins
  # on it, which strips away debug and arch information.
  process_bc(devicelib-${arch}.bc
    LIB_TGT builtins_${arch}.opt
    IN_FILE ${builtins_link_lib_${arch}}
    OUT_DIR ${bc_binary_dir}
    OPT_FLAGS ${opt_flags_${arch}}
    DEPENDENCIES device_lib_device_${arch})
  add_dependencies(libsycldevice-bc prepare-devicelib-${arch}.bc)
  set(complete_${arch}_libdev
    $<TARGET_PROPERTY:prepare-devicelib-${arch}.bc,TARGET_FILE>)
  install( FILES ${complete_${arch}_libdev}
           DESTINATION ${install_dest_bc}
           COMPONENT libsycldevice)
endforeach()

# Add host device imf libraries for obj and new offload objects.
foreach(dtype IN ITEMS bf16 fp32 fp64)
  foreach(ftype IN ITEMS obj obj-new-offload)
    set(tgt_name imf_fallback_${dtype}_host_${ftype})

    add_lib_imf(fallback-imf-${dtype}-host
      DIR ${${ftype}_binary_dir}
      FTYPE ${ftype}
      DTYPE ${dtype}
      EXTRA_OPTS ${${ftype}_host_compile_opts}
      TGT_NAME ${tgt_name})

    set(wrapper_name imf_wrapper.cpp)
    if (NOT ("${dtype}" STREQUAL "fp32"))
      set(wrapper_name imf_wrapper_${dtype}.cpp)
    endif()
    add_custom_command(
      OUTPUT ${${ftype}_binary_dir}/imf-${dtype}-host.${${ftype}-suffix}
      COMMAND ${clang_exe} ${${ftype}_host_compile_opts}
              ${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_name}
              -o ${${ftype}_binary_dir}/imf-${dtype}-host.${${ftype}-suffix}
      MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${wrapper_name}
      DEPENDS ${imf_obj_deps}
      VERBATIM)

    add_custom_target(imf_${dtype}_host_${ftype} DEPENDS
      ${obj_binary_dir}/imf-${dtype}-host.${${ftype}-suffix})
  endforeach()
endforeach()

foreach(ftype IN ITEMS obj obj-new-offload)
  add_custom_target(imf_host_${ftype}
    DEPENDS ${${ftype}_binary_dir}/${devicelib_host_static_${ftype}})
  add_custom_command(
    OUTPUT ${${ftype}_binary_dir}/${devicelib_host_static_${ftype}}
    COMMAND ${llvm-ar_exe} rcs
            ${${ftype}_binary_dir}/${devicelib_host_static_${ftype}}
            ${${ftype}_binary_dir}/imf-fp32-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/fallback-imf-fp32-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/imf-fp64-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/fallback-imf-fp64-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/imf-bf16-host.${${ftype}-suffix}
            ${${ftype}_binary_dir}/fallback-imf-bf16-host.${${ftype}-suffix}
    DEPENDS imf_fp32_host_${ftype} imf_fallback_fp32_host_${ftype}
    DEPENDS imf_fp64_host_${ftype} imf_fallback_fp64_host_${ftype}
    DEPENDS imf_bf16_host_${ftype} imf_fallback_bf16_host_${ftype}
    DEPENDS ${llvm-ar_target}
    VERBATIM)
  add_dependencies(libsycldevice-obj imf_host_${ftype})

  install( FILES ${obj_binary_dir}/${devicelib_host_static_${ftype}}
           DESTINATION ${install_dest_obj}
           COMPONENT libsycldevice)
endforeach()

foreach(ftype IN LISTS filetypes)
  install(
    FILES ${${ftype}_binary_dir}/libsycl-fallback-imf.${${ftype}-suffix}
          ${${ftype}_binary_dir}/libsycl-fallback-imf-fp64.${${ftype}-suffix}
          ${${ftype}_binary_dir}/libsycl-fallback-imf-bf16.${${ftype}-suffix}
    DESTINATION ${install_dest_${ftype}}
    COMPONENT libsycldevice)
endforeach()

