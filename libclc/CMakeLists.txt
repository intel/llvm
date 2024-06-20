cmake_minimum_required(VERSION 3.20.0)

project( libclc VERSION 0.2.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

# Add path for custom modules
list( INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/modules" )

set( LIBCLC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( LIBCLC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )
set( LIBCLC_OBJFILE_DIR ${LIBCLC_BINARY_DIR}/obj.libclc.dir )

include( AddLibclc )

include( GNUInstallDirs )
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  amdgcn-amdhsa/lib/SOURCES;
  amdgcn-amdhsa/libspirv/SOURCES;
  amdgcn/lib/SOURCES;
  amdgcn/libspirv/SOURCES;
  amdgcn-mesa3d/lib/SOURCES;
  amdgpu/lib/SOURCES;
  clspv/lib/SOURCES;
  clspv64/lib/SOURCES;
  generic/lib/SOURCES;
  generic/libspirv/SOURCES;
  ptx/lib/SOURCES;
  ptx-nvidiacl/lib/SOURCES;
  ptx-nvidiacl/libspirv/SOURCES;
  r600/lib/SOURCES;
  r600/libspirv/SOURCES;
  spirv/lib/SOURCES;
  spirv64/lib/SOURCES
  native_cpu-unknown-linux/libspirv/SOURCES
)

set( LIBCLC_MIN_LLVM 3.9.0 )

set( LIBCLC_TARGETS_TO_BUILD "all"
    CACHE STRING "Semicolon-separated list of libclc targets to build, or 'all'." )

option( ENABLE_RUNTIME_SUBNORMAL "Enable runtime linking of subnormal support." OFF )

set( LIBCLC_NATIVECPU_FLAGS_X86_64 ""
  CACHE STRING "Semicolon-separated list of compiler flags for x86_64 libclc target.")

if( LIBCLC_STANDALONE_BUILD OR CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  # Out-of-tree configuration
  set( LIBCLC_STANDALONE_BUILD TRUE )

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  message( STATUS "libclc LLVM version: ${LLVM_PACKAGE_VERSION}" )

  if( LLVM_PACKAGE_VERSION VERSION_LESS LIBCLC_MIN_LLVM )
    message( FATAL_ERROR "libclc needs at least LLVM ${LIBCLC_MIN_LLVM}" )
  endif()

  # Import required tools as targets
  if( NOT EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
    foreach( tool IN ITEMS clang llvm-as llvm-link opt )
      find_program( LLVM_TOOL_${tool} ${tool} PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )
      add_executable( libclc::${tool} IMPORTED GLOBAL )
      set_target_properties( libclc::${tool} PROPERTIES IMPORTED_LOCATION ${LLVM_TOOL_${tool}} )
    endforeach()
  endif()
else()
  # In-tree configuration
  set( LIBCLC_STANDALONE_BUILD FALSE )

  set( LLVM_PACKAGE_VERSION ${LLVM_VERSION} )

  # Note that we check this later (for both build types) but we can provide a
  # more useful error message when built in-tree. We assume that LLVM tools are
  # always available so don't warn here.
  if( NOT clang IN_LIST LLVM_ENABLE_PROJECTS )
    message(FATAL_ERROR "Clang is not enabled, but is required to build libclc in-tree")
  endif()

  if( NOT EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
    foreach( tool IN ITEMS clang llvm-as llvm-link opt )
      add_executable(libclc::${tool} ALIAS ${tool})
    endforeach()
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
    add_executable( libclc::${tool} IMPORTED GLOBAL )
    set_target_properties( libclc::${tool} PROPERTIES
      IMPORTED_LOCATION ${LLVM_CUSTOM_TOOL_${tool}} )
  endforeach()

  # If we've requested a custom binary directory, there are some otherwise
  # optional tools which we want to ensure are present.
  if( NOT TARGET libclc::llvm-spirv OR NOT TARGET libclc::libclc-remangler )
    message( FATAL_ERROR "libclc toolchain incomplete!" )
  endif()
endif()

foreach( tool IN ITEMS clang opt llvm-as llvm-link )
  if( NOT TARGET libclc::${tool} )
    message( FATAL_ERROR "libclc toolchain incomplete - missing tool ${tool}!" )
  endif()
endforeach()

# llvm-spirv is an optional dependency, used to build spirv-* targets.
find_program( LLVM_SPIRV llvm-spirv PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH )

if( LLVM_SPIRV AND NOT TARGET libclc::llvm-spirv )
  add_executable( libclc::llvm-spirv IMPORTED GLOBAL )
  set_target_properties( libclc::llvm-spirv PROPERTIES IMPORTED_LOCATION ${LLVM_SPIRV} )
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
if( TARGET libclc::llvm-spirv )
  list( APPEND LIBCLC_TARGETS_ALL  spirv-mesa3d- spirv64-mesa3d- )
endif()

option( LIBCLC_GENERATE_REMANGLED_VARIANTS
  "Generate remangled variants of enabled libclc targets." OFF )

if( LIBCLC_TARGETS_TO_BUILD STREQUAL "all" )
  set( LIBCLC_TARGETS_TO_BUILD ${LIBCLC_TARGETS_ALL} )
endif()

option( LIBCLC_NATIVECPU_HOST_TARGET "Build libclc for Native CPU using the host triple." Off)

if( LIBCLC_NATIVECPU_HOST_TARGET )
  list(APPEND LIBCLC_TARGETS_TO_BUILD ${LLVM_TARGET_TRIPLE})
endif()

list( SORT LIBCLC_TARGETS_TO_BUILD )

# Verify that the user hasn't requested mesa3d targets without an available
# llvm-spirv tool.
if( "spirv-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD OR "spirv64-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD )
  if( NOT TARGET libclc::llvm-spirv )
    message( FATAL_ERROR "SPIR-V targets requested, but spirv-tools is not installed" )
  endif()
endif()

add_custom_target(libspirv-builtins COMMENT "Build libspirv builtins")
add_custom_target(libclc-builtins COMMENT "Build libclc builtins")

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
# TODO: Does this need to be set for each possible triple?
set( x86_64-unknown-linux-gnu_devices none )
set( aarch64-unknown-linux-gnu_devices none )

# Setup aliases
set( cedar_aliases palm sumo sumo2 redwood juniper )
set( cypress_aliases hemlock )
set( barts_aliases turks caicos )
set( cayman_aliases aruba )
set( tahiti_aliases pitcairn verde oland hainan bonaire kabini kaveri hawaii
  mullins tonga tongapro iceland carrizo fiji stoney polaris10 polaris11
  gfx602 gfx705 gfx805
  gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx940 gfx941 gfx942
  gfx1010 gfx1011 gfx1012 gfx1013
  gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
  gfx1100 gfx1101 gfx1102 gfx1103
  gfx1150 gfx1151 gfx1152
  gfx1200 gfx1201
)

# pkg-config file
configure_file( libclc.pc.in libclc.pc @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/libclc.pc DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig" )
install( DIRECTORY generic/include/clc DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )

if( ENABLE_RUNTIME_SUBNORMAL )
  foreach( file subnormal_use_default subnormal_disable )
    link_bc(
       TARGET ${file}
       INPUTS ${PROJECT_SOURCE_DIR}/generic/lib/${file}.ll
    )
    install( FILES $<TARGET_PROPERTY:${file},TARGET_FILE> ARCHIVE
      DESTINATION "${CMAKE_INSTALL_DATADIR}/clc" )
  endforeach()
endif()

find_package( Python3 REQUIRED COMPONENTS Interpreter )
file( TO_CMAKE_PATH ${PROJECT_SOURCE_DIR}/generic/lib/gen_convert.py clc_script_loc )
file( TO_CMAKE_PATH ${PROJECT_SOURCE_DIR}/generic/libspirv/gen_core_convert.py core_script_loc )
file( TO_CMAKE_PATH ${PROJECT_SOURCE_DIR}/generic/libspirv/gen_convert.py spirv_script_loc )
file( TO_CMAKE_PATH ${PROJECT_SOURCE_DIR}/generic/lib/gen_convert.py script_loc )
add_custom_command(
  OUTPUT convert.cl
  COMMAND ${Python3_EXECUTABLE} ${script_loc} > convert.cl
  DEPENDS ${script_loc} )
add_custom_target( "generate_convert.cl" DEPENDS convert.cl )

add_custom_command(
  OUTPUT convert-core.cl
  COMMAND ${Python3_EXECUTABLE} ${core_script_loc} > convert-core.cl
  DEPENDS ${core_script_loc} )
add_custom_target( "generate_convert_core.cl" DEPENDS convert-core.cl )

add_custom_command(
  OUTPUT convert-spirv.cl
  COMMAND ${Python3_EXECUTABLE} ${spirv_script_loc} > convert-spirv.cl
  DEPENDS ${spirv_script_loc} )
add_custom_target( "generate_convert_spirv.cl" DEPENDS convert-spirv.cl )

add_custom_command(
  OUTPUT convert-clc.cl
  COMMAND ${Python3_EXECUTABLE} ${clc_script_loc} > convert-clc.cl
  DEPENDS ${clc_script_loc} )
add_custom_target( "generate_convert_clc.cl" DEPENDS convert-clc.cl )

add_custom_command(
  OUTPUT clspv-convert.cl
  COMMAND ${Python3_EXECUTABLE} ${script_loc} --clspv > clspv-convert.cl
  DEPENDS ${script_loc} )
add_custom_target( "clspv-generate_convert.cl" DEPENDS clspv-convert.cl )

enable_testing()

if (LIBCLC_STANDALONE_BUILD)
  set(LIBCLC_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
else(LIBCLC_STANDALONE_BUILD)
  set(LIBCLC_LIBRARY_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
endif(LIBCLC_STANDALONE_BUILD)
file( TO_CMAKE_PATH ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/clc LIBCLC_LIBRARY_OUTPUT_INTDIR )

set(NATIVECPU_SUPPORTED_ARCH "x86_64;aarch64")

foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
  message( STATUS "libclc target '${t}' is enabled" )
  string( REPLACE "-" ";" TRIPLE  ${t} )
  list( GET TRIPLE 0 ARCH )
  list( GET TRIPLE 1 VENDOR )
  list( GET TRIPLE 2 OS )

  set( dirs )

  if ( NOT ${ARCH} STREQUAL spirv AND NOT ${ARCH} STREQUAL spirv64 AND
       NOT ${ARCH} STREQUAL clspv AND NOT ${ARCH} STREQUAL clspv64)
    LIST( APPEND dirs generic )
  endif()

  if( ${ARCH} STREQUAL r600 OR ${ARCH} STREQUAL amdgcn )
    list( APPEND dirs amdgpu )
  endif()

  # nvptx is special
  if( ${ARCH} STREQUAL nvptx OR ${ARCH} STREQUAL nvptx64 )
    set( DARCH ptx )
  else()
    set( DARCH ${ARCH} )
  endif()

  set(IS_NATIVE_CPU_ARCH FALSE)
  if( ARCH IN_LIST NATIVECPU_SUPPORTED_ARCH )
    set(IS_NATIVE_CPU_ARCH TRUE)
  endif()

  if( IS_NATIVE_CPU_ARCH AND OS STREQUAL linux)
    LIST( APPEND dirs native_cpu-unknown-linux )
  elseif( IS_NATIVE_CPU_ARCH AND NOT OS STREQUAL linux )
    message(WARNING "libclc is being built for an unsupported ARCH/OS"
      " configuration, some SYCL programs may fail to build.")
  endif()

  set( lib_files )
  set( lib_gen_files )
  libclc_configure_lib_source(lib_files lib_gen_files
    LIB_DIR lib
    DIRS ${dirs} ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS}
    DEPS convert-clc.cl )

  set( libspirv_files )
  set( libspirv_gen_files )
  libclc_configure_lib_source(libspirv_files libspirv_gen_files
    LIB_DIR libspirv
    DIRS ${dirs} ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS}
    DEPS convert-spirv.cl convert-core.cl)

  foreach( d ${${t}_devices} )
    get_libclc_device_info(
      TRIPLE ${t}
      DEVICE ${d}
      CPU cpu
      ARCH_SUFFIX arch_suffix
      CLANG_TRIPLE clang_triple
    )

    # Some targets don't have a specific GPU to target
    set( flags )
    if( d STREQUAL none OR ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
      # FIXME: Ideally we would not be tied to a specific PTX ISA version
      if( ARCH STREQUAL nvptx OR ARCH STREQUAL nvptx64 )
        # Disables NVVM reflection to defer to after linking
        list( APPEND flags -Xclang -target-feature -Xclang +ptx72
             -march=sm_86 -mllvm --nvvm-reflect-enable=false)
      elseif( ARCH STREQUAL amdgcn )
        # AMDGCN needs libclc to be compiled to high bc version since all atomic
        # clang builtins need to be accessible
        list( APPEND flags -mcpu=gfx940 -mllvm --amdgpu-oclc-reflect-enable=false )
      elseif( IS_NATIVE_CPU_ARCH )
        list( APPEND flags -Xclang -fsycl-is-native-cpu )
        if( ARCH  STREQUAL x86_64 )
          list( APPEND flags ${LIBCLC_NATIVECPU_FLAGS_X86_64})
        endif()
      endif()
    endif()

    if( NOT "${cpu}" STREQUAL "" )
      list( APPEND flags -mcpu=${cpu} )
    endif()

    message( STATUS "  device: ${d} ( ${${d}_aliases} )" )

    # Note: when declaring builtins, we must consider that even if a target
    # formally/nominally supports the generic address space, in practice that
    # target may map it to the same target address space as another address
    # space (often the private one). In such cases we must be careful not to
    # multiply-define a builtin in a single target address space, as it would
    # result in a mangling clash.
    # For this reason we must consider the target support of the generic
    # address space separately from the *implementation* decision about whether
    # to declare certain builtins in that address space.
    set ( supports_generic_addrspace TRUE )
    # Note: we assume that if there is no distinct generic address space, it
    # maps to the private address space.
    set ( has_distinct_generic_addrspace TRUE )
    if ( ARCH STREQUAL spirv OR ARCH STREQUAL spirv64 )
      set( opt_flags -O3 )
      set( spvflags --spirv-max-version=1.1 )
    elseif( ARCH STREQUAL clspv OR ARCH STREQUAL clspv64 )
      set( opt_flags -O3 )
    elseif( ARCH STREQUAL nvptx OR ARCH STREQUAL nvptx64 )
      set( opt_flags -O3 "--nvvm-reflect-enable=false" )
      set( has_distinct_generic_addrspace FALSE )
    elseif( ARCH STREQUAL amdgcn )
      set( opt_flags -O3 --amdgpu-oclc-reflect-enable=false )
    elseif( IS_NATIVE_CPU_ARCH )
      set( opt_flags -O3 )
      set( has_distinct_generic_addrspace FALSE )
    else()
      set( opt_flags -O3 )
    endif()

    # Enable SPIR-V builtin function declarations, so they don't
    # have to be explicity declared in the soruce.
    list( APPEND flags -Xclang -fdeclare-spirv-builtins)

    set( LIBCLC_ARCH_OBJFILE_DIR "${LIBCLC_OBJFILE_DIR}/${arch_suffix}" )
    file( MAKE_DIRECTORY ${LIBCLC_ARCH_OBJFILE_DIR} )

    # OpenCL 3.0 extensions
    string(CONCAT CL_3_0_EXTENSIONS
      "-cl-ext="
      "+cl_khr_fp64,"
      "+cl_khr_fp16,"
      "+__opencl_c_3d_image_writes,"
      "+__opencl_c_images,"
      "+cl_khr_3d_image_writes")
    if( ARCH STREQUAL "aarch64")
      string( APPEND CL_3_0_EXTENSIONS ",+cl_clang_storage_class_specifiers,+__opencl_c_fp64,+cl_khr_int64_base_atomics" )
    endif()
    if( supports_generic_addrspace )
      string( APPEND CL_3_0_EXTENSIONS ",+__opencl_c_generic_address_space" )
      if( has_distinct_generic_addrspace )
        list( APPEND flags -D__CLC_DISTINCT_GENERIC_ADDRSPACE__ )
      endif()
    else()
      # Explictly disable opencl_c_generic_address_space (it may be enabled
      # by default on some targets). We also disable opencl_c_pipes and
      # opencl_c_device_enqueue since they require generic address space.
      string( APPEND CL_3_0_EXTENSIONS ",-__opencl_c_generic_address_space" )
      string( APPEND CL_3_0_EXTENSIONS ",-__opencl_c_pipes" )
      string( APPEND CL_3_0_EXTENSIONS ",-__opencl_c_device_enqueue" )
    endif()
    list( APPEND flags -cl-std=CL3.0 "-Xclang" ${CL_3_0_EXTENSIONS} )

    # Add platform specific flags
    if(WIN32)
      list(APPEND flags -D_WIN32)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      list(APPEND flags -D__APPLE__)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      list(APPEND flags -D__unix__ -D__linux__)
    else()
      # Assume some UNIX system otherwise
      list(APPEND flags -D__unix__)
    endif()

    add_libclc_builtin_set(libspirv-${arch_suffix}
      TRIPLE ${clang_triple}
      TARGET_ENV libspirv
      COMPILE_OPT ${flags}
      OPT_FLAGS ${opt_flags}
      FILES ${libspirv_files}
      GEN_FILES ${libspirv_gen_files}
      ALIASES ${${d}_aliases}
      GENERATE_TARGET "generate_convert_spirv.cl" "generate_convert_core.cl"
      PARENT_TARGET libspirv-builtins)

    add_libclc_builtin_set(clc-${arch_suffix}
      TRIPLE ${clang_triple}
      TARGET_ENV clc
      COMPILE_OPT ${flags}
      OPT_FLAGS ${opt_flags}
      FILES ${lib_files}
      GEN_FILES ${lib_gen_files}
      LIB_DEP libspirv-${arch_suffix}
      ALIASES ${${d}_aliases}
      GENERATE_TARGET "generate_convert_clc.cl"
      PARENT_TARGET libclc-builtins)
  endforeach( d )
endforeach( t )

install(DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  COMPONENT libspirv-builtins
  FILES_MATCHING PATTERN "libspirv-*")

install(DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  COMPONENT clc-builtins
  FILES_MATCHING PATTERN "clc-*")

if( LIBCLC_GENERATE_REMANGLED_VARIANTS )
  install(DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
    DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    COMPONENT libspirv-builtins
    FILES_MATCHING PATTERN "remangled-*libspirv-*")

  install(DIRECTORY ${LIBCLC_LIBRARY_OUTPUT_INTDIR}
    DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    COMPONENT clc-builtins
    FILES_MATCHING PATTERN "remangled-*clc-*")
endif()

add_subdirectory(test)
