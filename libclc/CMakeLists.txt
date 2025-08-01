cmake_minimum_required(VERSION 3.20.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libclc VERSION 0.2.0 LANGUAGES CXX C)
endif()
set(LLVM_SUBPROJECT_TITLE "libclc")

set(CMAKE_CXX_STANDARD 17)

# Add path for custom modules
list( INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules" )

set( LIBCLC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( LIBCLC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )
set( LIBCLC_OBJFILE_DIR ${LIBCLC_BINARY_DIR}/obj.libclc.dir )

include( AddLibclc )

include( GNUInstallDirs )
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  # OpenCL libraries
  opencl/lib/amdgcn-amdhsa/SOURCES;
  opencl/lib/amdgcn/SOURCES;
  opencl/lib/clspv/SOURCES;
  opencl/lib/generic/SOURCES;
  opencl/lib/ptx-nvidiacl/SOURCES;
  opencl/lib/r600/SOURCES;
  opencl/lib/spirv/SOURCES;
  # CLC internal libraries
  clc/lib/generic/SOURCES;
  clc/lib/amdgcn/SOURCES;
  clc/lib/amdgpu/SOURCES;
  clc/lib/clspv/SOURCES;
  clc/lib/r600/SOURCES;
  clc/lib/spirv/SOURCES;
  # libspirv libraries
  libspirv/lib/generic/SOURCES;
  libspirv/lib/amdgcn-amdhsa/SOURCES;
  libspirv/lib/amdgcn/SOURCES;
  libspirv/lib/ptx-nvidiacl/SOURCES;
  libspirv/lib/r600/SOURCES;
  libspirv/lib/native_cpu/SOURCES;
)

set( LIBCLC_MIN_LLVM 3.9.0 )

set( LIBCLC_TARGETS_TO_BUILD "all"
    CACHE STRING "Semicolon-separated list of libclc targets to build, or 'all'." )

option( ENABLE_RUNTIME_SUBNORMAL "Enable runtime linking of subnormal support." OFF )

if( LIBCLC_STANDALONE_BUILD OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  # Out-of-tree configuration
  set( LIBCLC_STANDALONE_BUILD TRUE )

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  message( STATUS "libclc LLVM version: ${LLVM_PACKAGE_VERSION}" )

  if( LLVM_PACKAGE_VERSION VERSION_LESS LIBCLC_MIN_LLVM )
    message( FATAL_ERROR "libclc needs at least LLVM ${LIBCLC_MIN_LLVM}" )
  endif()

  # Import required tools
  if( NOT EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
    foreach( tool IN ITEMS clang llvm-as llvm-link llvm-spirv opt )
      find_program( LLVM_TOOL_${tool} ${tool} PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )
      set( ${tool}_exe ${LLVM_TOOL_${tool}} )
      set( ${tool}_target )
    endforeach()
  endif()
else()
  # In-tree configuration
  set( LIBCLC_STANDALONE_BUILD FALSE )

  set( LLVM_PACKAGE_VERSION ${LLVM_VERSION} )

  # Note that we check this later (for both build types) but we can provide a
  # more useful error message when built in-tree. We assume that LLVM tools are
  # always available so don't warn here.
  if( NOT LLVM_RUNTIMES_BUILD AND NOT clang IN_LIST LLVM_ENABLE_PROJECTS )
    message(FATAL_ERROR "Clang is not enabled, but is required to build libclc in-tree")
  endif()

  if( NOT EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
    get_host_tool_path( clang CLANG clang_exe clang_target )
    get_host_tool_path( llvm-as LLVM_AS llvm-as_exe llvm-as_target )
    get_host_tool_path( llvm-link LLVM_LINK llvm-link_exe llvm-link_target )
    get_host_tool_path( llvm-spirv LLVM_SPIRV llvm-spirv_exe llvm-spirv_target )
    get_host_tool_path( opt OPT opt_exe opt_target )
  endif()
endif()

if( EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
  message( WARNING "Using custom LLVM tools to build libclc: "
    "${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR}, "
    " ensure the tools are up to date." )
  # Note - use a differently named variable than LLVM_TOOL_${tool} as above, as
  # the variable name is used to cache the result of find_program. If we used
  # the same name, a user wouldn't be able to switch a build between default
  # and custom tools.
  foreach( tool IN ITEMS clang llvm-as llvm-link opt llvm-spirv libclc-remangler )
    find_program( LLVM_CUSTOM_TOOL_${tool} ${tool}
      PATHS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )
    set( ${tool}_exe ${LLVM_CUSTOM_TOOL_${tool}} )
    set( ${tool}_target )
  endforeach()

  # If we've requested a custom binary directory, there are some otherwise
  # optional tools which we want to ensure are present.
  foreach( tool IN ITEMS llvm-spirv libclc-remangler )
    if( NOT EXISTS "${${tool}_exe}" AND "${${tool}_target}" STREQUAL "" )
      message( FATAL_ERROR "libclc toolchain incomplete!" )
    endif()
  endforeach()
endif()

foreach( tool IN ITEMS clang opt llvm-as llvm-link )
  if( NOT EXISTS "${${tool}_exe}" AND "${${tool}_target}" STREQUAL "" )
    message( FATAL_ERROR "libclc toolchain incomplete - missing tool ${tool}!" )
  endif()
endforeach()

# llvm-spirv is an optional dependency, used to build spirv-* targets.
# It may be provided in-tree or externally.
if( TARGET llvm-spirv )
  get_host_tool_path( llvm-spirv LLVM_SPIRV llvm-spirv_exe llvm-spirv_target )
else()
  find_program( LLVM_SPIRV llvm-spirv HINTS ${LLVM_TOOLS_BINARY_DIR} )
  set( llvm-spirv_exe "${LLVM_SPIRV}" )
  set( llvm-spirv_target )
endif()

# List of all targets. Note that some are added dynamically below.
set( LIBCLC_TARGETS_ALL
  amdgcn--
  amdgcn--amdhsa
  clspv--
  clspv64--
  r600--
  nvptx--
  nvptx64--
  nvptx--nvidiacl
  nvptx64--nvidiacl
)
set( LIBCLC_TEST_TARGETS_ALL
  nvptx--nvidiacl
  nvptx64--nvidiacl
  amdgcn--amdhsa
)

# mesa3d environment is only available since LLVM 4.0
if( LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL 4.0.0 )
  list( APPEND LIBCLC_TARGETS_ALL amdgcn-mesa-mesa3d )
endif()

# spirv-mesa3d and spirv64-mesa3d targets can only be built with the (optional)
# llvm-spirv external tool.
if( llvm-spirv_exe )
  list( APPEND LIBCLC_TARGETS_ALL  spirv-mesa3d- spirv64-mesa3d- )
endif()

option( LIBCLC_GENERATE_REMANGLED_VARIANTS
  "Generate remangled variants of enabled libclc targets." OFF )

# Verify that the user hasn't requested mesa3d targets without an available
# llvm-spirv tool.
if( "spirv-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD OR "spirv64-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD )
  if( NOT llvm-spirv_exe )
    message( FATAL_ERROR "SPIR-V targets requested, but spirv-tools is not installed" )
  endif()
endif()

add_custom_target(libspirv-builtins COMMENT "Build libspirv builtins")
add_custom_target(libopencl-builtins COMMENT "Build libclc builtins")

if( LIBCLC_TARGETS_TO_BUILD STREQUAL "all" )
  set( LIBCLC_TARGETS_TO_BUILD ${LIBCLC_TARGETS_ALL} )
else()
  foreach(TARGET_TO_BUILD ${LIBCLC_TARGETS_TO_BUILD})
    if (NOT ${TARGET_TO_BUILD} IN_LIST LIBCLC_TARGETS_ALL)
      message ( FATAL_ERROR "Unknown target in LIBCLC_TARGETS_TO_BUILD: \"${TARGET_TO_BUILD}\"\n"
                            "Valid targets are: ${LIBCLC_TARGETS_ALL}\n")
    endif()
  endforeach()
endif()

option( LIBCLC_NATIVECPU_HOST_TARGET "Build libclc for Native CPU." Off)

if( LIBCLC_NATIVECPU_HOST_TARGET )
  list(APPEND LIBCLC_TARGETS_TO_BUILD native_cpu)
endif()

list( SORT LIBCLC_TARGETS_TO_BUILD )

# Construct LLVM version define
set( LLVM_VERSION_DEFINE "-DHAVE_LLVM=0x${LLVM_VERSION_MAJOR}0${LLVM_VERSION_MINOR}" )

set(LIBCLC_TARGET_TO_TEST)

foreach ( t ${LIBCLC_TEST_TARGETS_ALL})
  if( t IN_LIST LIBCLC_TARGETS_TO_BUILD )
    list( APPEND LIBCLC_TARGET_TO_TEST "${t}" )
  endif()
endforeach(t)

# Configure prepare_builtins
add_subdirectory(utils)

# Setup arch devices
set( r600--_devices cedar cypress barts cayman )
set( amdgcn--_devices tahiti )
set( amdgcn-mesa-mesa3d_devices ${amdgcn--_devices} )
set( amdgcn--amdhsa_devices none )
set( clspv--_devices none )
set( clspv64--_devices none )
set( nvptx--_devices none )
set( nvptx64--_devices none )
set( nvptx--nvidiacl_devices none )
set( nvptx64--nvidiacl_devices none )
set( spirv-mesa3d-_devices none )
set( spirv64-mesa3d-_devices none )
set( native_cpu_devices none )

# Setup aliases
set( cedar_aliases palm sumo sumo2 redwood juniper )
set( cypress_aliases hemlock )
set( barts_aliases turks caicos )
set( cayman_aliases aruba )
set( tahiti_aliases pitcairn verde oland hainan bonaire kabini kaveri hawaii
  mullins tonga tongapro iceland carrizo fiji stoney polaris10 polaris11
  gfx602 gfx705 gfx805
  gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx942 gfx950
  gfx1010 gfx1011 gfx1012 gfx1013
  gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
  gfx1100 gfx1101 gfx1102 gfx1103
  gfx1150 gfx1151 gfx1152 gfx1153
  gfx1200 gfx1201
)

# pkg-config file
configure_file( libclc.pc.in libclc.pc @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/libclc.pc DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig" )

if( ENABLE_RUNTIME_SUBNORMAL )
  foreach( file IN ITEMS subnormal_use_default subnormal_disable )
    link_bc(
       TARGET ${file}
       RSP_DIR ${LIBCLC_ARCH_OBJFILE_DIR}
       INPUTS ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/${file}.ll
    )
    install(
      FILES $<TARGET_PROPERTY:${file},TARGET_FILE>
      DESTINATION "${CMAKE_INSTALL_DATADIR}/clc"
    )
  endforeach()
endif()

find_package( Python3 REQUIRED COMPONENTS Interpreter )
file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/utils/gen_convert.py script_loc )
file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libspirv/lib/generic/gen_convert.py spirv_script_loc )
file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libspirv/lib/generic/gen_convert_common.py spirv_common_script_loc )
add_custom_command(
  OUTPUT convert.cl
  COMMAND ${Python3_EXECUTABLE} ${script_loc} > convert.cl
  DEPENDS ${script_loc} )
add_custom_target( generate-convert.cl DEPENDS convert.cl )
set_target_properties( generate-convert.cl PROPERTIES FOLDER "libclc/Sourcegenning" )

add_custom_command(
  OUTPUT clc-convert.cl
  COMMAND ${Python3_EXECUTABLE} ${script_loc} --clc > clc-convert.cl
  DEPENDS ${script_loc} )
add_custom_target( generate-clc-convert.cl DEPENDS clc-convert.cl )
set_target_properties( generate-clc-convert.cl PROPERTIES FOLDER "libclc/Sourcegenning" )

add_custom_command(
  OUTPUT spirv-convert.cl
  COMMAND ${Python3_EXECUTABLE} ${spirv_script_loc} > spirv-convert.cl
  DEPENDS ${spirv_script_loc} ${spirv_common_script_loc} )
add_custom_target( generate-spirv-convert.cl DEPENDS spirv-convert.cl )
set_target_properties( generate-spirv-convert.cl PROPERTIES FOLDER "libclc/Sourcegenning" )

if ( clspv-- IN_LIST LIBCLC_TARGETS_TO_BUILD OR clspv64-- IN_LIST LIBCLC_TARGETS_TO_BUILD )
  add_custom_command(
    OUTPUT clspv-convert.cl
    COMMAND ${Python3_EXECUTABLE} ${script_loc} --clspv > clspv-convert.cl
    DEPENDS ${script_loc} )
  add_custom_target( generate-clspv-convert.cl DEPENDS clspv-convert.cl )
  set_target_properties( generate-clspv-convert.cl PROPERTIES FOLDER "libclc/Sourcegenning" )
endif()

set_source_files_properties(
  # CLC builtins
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_cos.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_divide.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_exp10.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_exp2.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_exp.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_log10.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_log2.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_log.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_powr.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_recip.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_rsqrt.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_sin.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_sqrt.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/generic/math/clc_native_tan.cl
  # Target-specific CLC builtins
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/amdgpu/math/clc_native_exp2.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/amdgpu/math/clc_native_exp.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/amdgpu/math/clc_native_log10.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/clc/lib/r600/math/clc_native_rsqrt.cl
  # OpenCL builtins
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_cos.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_divide.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_exp.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_exp10.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_exp2.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_log.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_log10.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_log2.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_powr.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_recip.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_rsqrt.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_sin.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_sqrt.cl
  ${CMAKE_CURRENT_SOURCE_DIR}/opencl/lib/generic/math/native_tan.cl
  PROPERTIES COMPILE_OPTIONS -fapprox-func
)

enable_testing()

if (LIBCLC_STANDALONE_BUILD)
  set(LIBCLC_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
else(LIBCLC_STANDALONE_BUILD)
  set(LIBCLC_LIBRARY_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
endif(LIBCLC_STANDALONE_BUILD)
file( TO_CMAKE_PATH ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/clc LIBCLC_LIBRARY_OUTPUT_INTDIR )

foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
  message( STATUS "libclc target '${t}' is enabled" )
  string( REPLACE "-" ";" TRIPLE  ${t}-- )
  list( GET TRIPLE 0 ARCH )
  list( GET TRIPLE 1 VENDOR )
  list( GET TRIPLE 2 OS )

  set( opencl_dirs )

  if( ${ARCH} STREQUAL r600 OR ${ARCH} STREQUAL amdgcn )
    list( APPEND opencl_dirs amdgpu )
  endif()

  # Some targets' directories alias others
  if( ${ARCH} STREQUAL nvptx OR ${ARCH} STREQUAL nvptx64 )
    set( DARCH ptx )
  elseif( ${ARCH} STREQUAL clspv OR ${ARCH} STREQUAL clspv64 )
    set( DARCH clspv )
  elseif( ${ARCH} STREQUAL spirv OR ${ARCH} STREQUAL spirv64 )
    set( DARCH spirv )
  elseif( ${ARCH} STREQUAL amdgcn-mesa3d )
    set( DARCH amdgcn-amdhsa )
  else()
    set( DARCH ${ARCH} )
  endif()

  # Append a variety of target- and triple-based directories to search,
  # increasing in specificity.
  list( APPEND opencl_dirs ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS} )

  # The 'generic' directory contains all of the generic implementations of the
  # builtins. It is included first so it has the lowest search priority,
  # allowing targets to override builtins based on file names found later in
  # the list of search directories.
  # CLC builds all builtins for all targets, so unconditionally prepend the
  # 'generic' directory.
  set( clc_dirs generic ${opencl_dirs} )
  # Some OpenCL targets don't build all builtins, in which case they don't want
  # the 'generic' directory. Otherwise, prepend the 'generic' directory.
  if ( NOT ARCH STREQUAL spirv AND NOT ARCH STREQUAL spirv64 AND
       NOT ARCH STREQUAL clspv AND NOT ARCH STREQUAL clspv64)
    list( PREPEND opencl_dirs generic )
  endif()

  set( libspirv_dirs ${opencl_dirs} )

  set( clc_lib_files )
  set( clc_gen_files clc-convert.cl )

  libclc_configure_lib_source(
    clc_lib_files
    LIB_ROOT_DIR clc
    DIRS ${clc_dirs}
  )

  set( opencl_lib_files )
  set( opencl_gen_files )

  if( NOT ARCH STREQUAL spirv AND NOT ARCH STREQUAL spirv64 )
    if( ARCH STREQUAL clspv OR ARCH STREQUAL clspv64 )
      list( APPEND opencl_gen_files clspv-convert.cl )
    else()
      list( APPEND opencl_gen_files convert.cl )
      if ( NOT ENABLE_RUNTIME_SUBNORMAL )
        list( APPEND opencl_lib_files opencl/lib/generic/subnormal_use_default.ll )
      endif()
    endif()
  endif()

  libclc_configure_lib_source(
    opencl_lib_files
    LIB_ROOT_DIR opencl
    DIRS ${opencl_dirs}
  )

  set( BUILD_LIBSPIRV_${t} FALSE )
  if ( t STREQUAL amdgcn--amdhsa OR t STREQUAL nvptx64--nvidiacl OR t STREQUAL native_cpu )
    set( libspirv_lib_files )
    set( libspirv_gen_files )
    set( BUILD_LIBSPIRV_${t} TRUE )

    if ( NOT ENABLE_RUNTIME_SUBNORMAL )
      list( APPEND libspirv_gen_files spirv-convert.cl )
      list( APPEND libspirv_lib_files libspirv/lib/generic/subnormal_use_default.ll )
    endif()

    libclc_configure_lib_source(
      libspirv_lib_files
      LIB_ROOT_DIR libspirv
      DIRS ${libspirv_dirs} ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS}
    )
  endif()

  foreach( d ${${t}_devices} )
    get_libclc_device_info(
      TRIPLE ${t}
      DEVICE ${d}
      CPU cpu
      ARCH_SUFFIX arch_suffix
      CLANG_TRIPLE clang_triple
    )

    message( STATUS "  device: ${d} ( ${${d}_aliases} )" )

    if ( ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
      set( build_flags -O0 -finline-hint-functions -DCLC_SPIRV )
      set( opt_flags )
      set( spvflags --spirv-max-version=1.1 )
      set( MACRO_ARCH SPIRV32 )
      if( ARCH STREQUAL spirv64 )
        set( MACRO_ARCH SPIRV64 )
      endif()
    elseif( ARCH STREQUAL clspv OR ARCH STREQUAL clspv64 )
      set( build_flags "-Wno-unknown-assumption" -DCLC_CLSPV )
      set( opt_flags -O3 )
      set( MACRO_ARCH CLSPV32 )
      if( ARCH STREQUAL clspv64 )
        set( MACRO_ARCH CLSPV64 )
      endif()
    elseif( ARCH STREQUAL nvptx OR ARCH STREQUAL nvptx64 )
      set( build_flags )
      set( opt_flags -O3 "--nvvm-reflect-enable=false" )
    elseif( ARCH STREQUAL amdgcn )
      set( build_flags )
      set( opt_flags -O3 --amdgpu-oclc-reflect-enable=false )
    else()
      set( build_flags )
      set( opt_flags -O3 )
      set( MACRO_ARCH ${ARCH} )
    endif()

    # Some targets don't have a specific GPU to target
    if( ARCH STREQUAL nvptx OR ARCH STREQUAL nvptx64 )
      # Disable NVVM reflection to deferring to after linking
      list( APPEND build_flags -Xclang -target-feature -Xclang +ptx72
           -march=sm_86 -mllvm --nvvm-reflect-enable=false )
    elseif( t STREQUAL amdgcn--amdhsa )
      # Disable AMDGPU reflection, deferring to after linking
      list( APPEND build_flags -mllvm --amdgpu-oclc-reflect-enable=false )
    endif()

    set( LIBCLC_ARCH_OBJFILE_DIR "${LIBCLC_OBJFILE_DIR}/${arch_suffix}" )
    file( MAKE_DIRECTORY ${LIBCLC_ARCH_OBJFILE_DIR} )

    # Build for OpenCL 3.0 independently of the target or device.
    list( APPEND build_flags -cl-std=CL3.0 )

    # OpenCL 3.0 extensions
    string(CONCAT CL_3_0_EXTENSIONS
      "-cl-ext="
      "+__opencl_c_3d_image_writes,"
      "+__opencl_c_images,"
      "+cl_khr_3d_image_writes")
    list( APPEND build_flags "-Xclang" ${CL_3_0_EXTENSIONS} )

    # Add platform specific flags
    if(WIN32)
      list(APPEND build_flags -D_WIN32)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      list(APPEND build_flags -D__APPLE__)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      list(APPEND build_flags -D__unix__ -D__linux__)
    else()
      # Assume some UNIX system otherwise
      list(APPEND build_flags -D__unix__)
    endif()

    string( TOUPPER "CLC_${MACRO_ARCH}" CLC_TARGET_DEFINE )

    list( APPEND build_flags
      -D${CLC_TARGET_DEFINE}
      # All libclc builtin libraries see CLC headers
      -I${CMAKE_CURRENT_SOURCE_DIR}/clc/include
      # Error on undefined macros
      -Werror=undef
    )

    if( NOT "${cpu}" STREQUAL "" )
      list( APPEND build_flags -mcpu=${cpu} )
    endif()

    # Generic address space support.
    # Note: when declaring builtins, we must consider that even if a target
    # formally/nominally supports the generic address space, in practice that
    # target may map it to the same target address space as another address
    # space (often the private one). In such cases we must be careful not to
    # multiply-define a builtin in a single target address space, as it would
    # result in a mangling clash.
    # For this reason we must consider the target support of the generic
    # address space separately from the *implementation* decision about whether
    # to declare certain builtins in that address space.
    # Note: we assume that if there is no distinct generic address space, it
    # maps to the private address space.
    set ( private_addrspace_val 0 )
    set ( generic_addrspace_val 0 )
    if( ARCH STREQUAL amdgcn OR ARCH STREQUAL r600 OR ARCH STREQUAL amdgcn-amdhsa )
      set ( private_addrspace_val 5 )
    endif()
    if( ARCH STREQUAL spirv OR ARCH STREQUAL spirv64)
      set ( generic_addrspace_val 4 )
    endif()
    list( APPEND build_flags
      -D__CLC_PRIVATE_ADDRSPACE_VAL=${private_addrspace_val}
      -D__CLC_GENERIC_ADDRSPACE_VAL=${generic_addrspace_val}
    )

    set( clc_build_flags ${build_flags} -DCLC_INTERNAL )

    add_libclc_builtin_set(
      CLC_INTERNAL
      ARCH ${ARCH}
      ARCH_SUFFIX clc-${arch_suffix}
      TRIPLE ${clang_triple}
      COMPILE_FLAGS ${clc_build_flags}
      OPT_FLAGS ${opt_flags}
      LIB_FILES ${clc_lib_files}
      GEN_FILES ${clc_gen_files}
    )

    if( BUILD_LIBSPIRV_${t} )
      set( spirv_build_flags ${build_flags} )
      list( APPEND spirv_build_flags
        # Enable SPIR-V builtin function declarations, so they don't have to be
        # explicity declared in the soruce.
        -Xclang -fdeclare-spirv-builtins
        -I${CMAKE_CURRENT_SOURCE_DIR}/opencl/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/libspirv/include/
      )

      add_libclc_builtin_set(
        ARCH ${ARCH}
        ARCH_SUFFIX libspirv-${arch_suffix}
        TRIPLE ${clang_triple}
        TARGET_ENV libspirv-
        REMANGLE ${LIBCLC_GENERATE_REMANGLED_VARIANTS}
        COMPILE_FLAGS ${spirv_build_flags}
        OPT_FLAGS ${opt_flags}
        LIB_FILES ${libspirv_lib_files}
        GEN_FILES ${libspirv_gen_files}
        ALIASES ${${d}_aliases}
        PARENT_TARGET libspirv-builtins
        # Link in the CLC builtins and internalize their symbols
        INTERNAL_LINK_DEPENDENCIES builtins.link.clc-${arch_suffix}
      )
    endif()

    set( opencl_build_flags ${build_flags} )
    list( APPEND opencl_build_flags
      -I${CMAKE_CURRENT_SOURCE_DIR}/opencl/include
    )

    add_libclc_builtin_set(
      ARCH ${ARCH}
      ARCH_SUFFIX ${arch_suffix}
      TRIPLE ${clang_triple}
      COMPILE_FLAGS ${opencl_build_flags}
      OPT_FLAGS ${opt_flags}
      LIB_FILES ${opencl_lib_files}
      GEN_FILES ${opencl_gen_files}
      ALIASES ${${d}_aliases}
      PARENT_TARGET libopencl-builtins
      # Link in the CLC builtins and internalize their symbols
      INTERNAL_LINK_DEPENDENCIES builtins.link.clc-${arch_suffix}
    )
  endforeach( d )
endforeach( t )

install(DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  COMPONENT libspirv-builtins
  FILES_MATCHING PATTERN "libspirv-*")

if( LIBCLC_GENERATE_REMANGLED_VARIANTS )
  install(DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
    DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    COMPONENT libspirv-builtins
    FILES_MATCHING PATTERN "remangled-*libspirv-*")
endif()

add_subdirectory(test)
