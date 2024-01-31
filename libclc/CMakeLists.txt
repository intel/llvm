cmake_minimum_required(VERSION 3.20.0)

add_custom_target(libspirv-builtins COMMENT "Build libspirv builtins")
add_custom_target(libclc-builtins COMMENT "Build libclc builtins")

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
  ${CMAKE_MODULE_PATH})

# If we are not building as a part of LLVM, build libclc as an
# standalone project, using LLVM/Clang as external tools.
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project( libclc VERSION 0.2.0 LANGUAGES CXX C )

  set( LIBCLC_STANDALONE_BUILD 1 )

  include(HandleOutOfTreeLLVM)
else()
  include(HandleInLLVMTree)
endif()

set(CMAKE_CXX_STANDARD 17)

include( GNUInstallDirs )
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  amdgcn-amdhsa/lib/SOURCES;
  amdgcn/lib/SOURCES;
  amdgcn-mesa3d/lib/SOURCES;
  amdgpu/lib/SOURCES;
  clspv/lib/SOURCES;
  clspv64/lib/SOURCES;
  generic/lib/SOURCES;
  ptx/lib/SOURCES;
  ptx-nvidiacl/lib/SOURCES;
  r600/lib/SOURCES;
  spirv/lib/SOURCES;
  spirv64/lib/SOURCES
)

include( AddLibclc )

# List of all targets
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
  spirv-mesa3d-
  spirv64-mesa3d-
)
set( LIBCLC_TEST_TARGETS_ALL
  nvptx--nvidiacl
  nvptx64--nvidiacl
  amdgcn--amdhsa
)

set( LIBCLC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set( LIBCLC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )

set( LIBCLC_TARGETS_TO_BUILD "all"
    CACHE STRING "Semicolon-separated list of targets to build, or 'all'." )

option( ENABLE_RUNTIME_SUBNORMAL "Enable runtime linking of subnormal support."
OFF )

option( LIBCLC_GENERATE_REMANGLED_VARIANTS
				"Generate remangled variants of enabled libclc targets."
				OFF )

# mesa3d environment is only available since LLVM 4.0
if( ${LLVM_VERSION} VERSION_GREATER "3.9.0" )
	set( LIBCLC_TARGETS_ALL ${LIBCLC_TARGETS_ALL} amdgcn-mesa-mesa3d )
endif()

if( LIBCLC_TARGETS_TO_BUILD STREQUAL "all" )
	set( LIBCLC_TARGETS_TO_BUILD ${LIBCLC_TARGETS_ALL} )
endif()

set(LIBCLC_TARGET_TO_TEST)

foreach ( t ${LIBCLC_TEST_TARGETS_ALL})
	if( ${t} IN_LIST LIBCLC_TARGETS_TO_BUILD )
		list( APPEND LIBCLC_TARGET_TO_TEST "${t}" )
	endif( ${t} IN_LIST LIBCLC_TARGETS_TO_BUILD )
endforeach(t)

execute_process( COMMAND ${LLVM_CONFIG} "--system-libs"
	OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
separate_arguments( LLVM_SYSTEM_LIBS )
execute_process( COMMAND ${LLVM_CONFIG} "--libs" "core" "bitreader" "bitwriter"
	OUTPUT_VARIABLE LLVM_LIBS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
separate_arguments( LLVM_LIBS )
execute_process( COMMAND ${LLVM_CONFIG} "--libdir"
	OUTPUT_VARIABLE LLVM_LIBDIR
	OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${LLVM_CONFIG} "--ldflags"
	OUTPUT_VARIABLE LLVM_LD_FLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
execute_process( COMMAND ${LLVM_CONFIG} "--cxxflags"
	OUTPUT_VARIABLE LLVM_CXX_FLAGS
	OUTPUT_STRIP_TRAILING_WHITESPACE )
separate_arguments( LLVM_CXX_FLAGS )
execute_process( COMMAND ${LLVM_CONFIG} "--bindir"
	OUTPUT_VARIABLE LLVM_BINDIR
	OUTPUT_STRIP_TRAILING_WHITESPACE )

# These were not properly reported in early LLVM and we don't need them
list( APPEND LLVM_CXX_FLAGS -fno-rtti -fno-exceptions )

# List containing all the toolchain variables.
list( APPEND BINARY_VARIABLES LLVM_CLANG LLVM_AS LLVM_LINK LLVM_OPT LLVM_SPIRV
      LIBCLC_REMANGLER )
# List containing all the names of toolchain binaries.
# NOTE: both lists (BINARY_VARIABLES and BINARY_NAMES) must be in sync.
list( APPEND BINARY_NAMES clang llvm-as llvm-link opt llvm-spirv
      libclc-remangler )

# find_program needs the variable to be cleared in order to perform a search.
# Make sure that the cached entries are cleared as well.
function( ClearVariables BINARY_VARIABLES_LIST )
  foreach( V ${BINARY_VARIABLES_LIST} )
    unset( ${V} CACHE )
    unset( ${V} PARENT_SCOPE )
  endforeach( V )
endfunction()

# Use find_program to locate toolchain binaries.
function( FindToolBinary BINARY_VARIABLES_LIST BINARY_NAMES_LIST PATH_NAME )
  list( LENGTH BINARY_NAMES_LIST COUNT )
  math( EXPR COUNT "${COUNT}-1" )
  foreach( I RANGE ${COUNT} )
    list( GET BINARY_VARIABLES_LIST ${I} BV )
    list( GET BINARY_NAMES_LIST ${I} BN )
    find_program( ${BV} ${BN} PATHS ${PATH_NAME} NO_DEFAULT_PATH )
  endforeach( I )
endfunction()

# Use custom toolchain to build libclc, this can be useful when dealing with
# debug builds, that do not need libclc bitcode to be built using debug tools.
if ( EXISTS ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )
  message( WARNING "Using custom LLVM tools to build libclc: "
    "${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR}, "
    " make sure that the tools are up to date." )

  # First clear the variables,
  ClearVariables( "${BINARY_VARIABLES}" )
  # then set.
  FindToolBinary( "${BINARY_VARIABLES}" "${BINARY_NAMES}"
    ${LIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR} )

  if( NOT LLVM_CLANG OR NOT LLVM_OPT OR NOT LLVM_AS OR NOT LLVM_LINK
      OR NOT LIBCLC_REMANGLER )
    message( FATAL_ERROR "Custom toolchain incomplete!" )
  endif()
endif()

# Print LLVM variables
message( "LLVM libdir: ${LLVM_LIBRARY_DIR}" )
message( "LLVM bindir: ${LLVM_TOOLS_BINARY_DIR}" )
message( "LLVM cxx flags: ${LLVM_CXX_FLAGS}" )
message( "" )

# It's OK to call find program again, if the variables have been set in the
# custom location clause, find_program returns immediately.
FindToolBinary( "${BINARY_VARIABLES}" "${BINARY_NAMES}"
  ${LLVM_TOOLS_BINARY_DIR} )

# Print toolchain
message( "clang: ${LLVM_CLANG}" )
message( "llvm-as: ${LLVM_AS}" )
message( "llvm-link: ${LLVM_LINK}" )
message( "opt: ${LLVM_OPT}" )
message( "llvm-spirv: ${LLVM_SPIRV}" )
message( "libclc-remangler: ${LIBCLC_REMANGLER}" )
message( "" )
if( NOT LLVM_CLANG OR NOT LLVM_OPT OR NOT LLVM_AS OR NOT LLVM_LINK
		OR NOT LIBCLC_REMANGLER )
	message( FATAL_ERROR "toolchain incomplete!" )
endif()

list( SORT LIBCLC_TARGETS_TO_BUILD )

if( "spirv-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD OR "spirv64-mesa3d-" IN_LIST LIBCLC_TARGETS_TO_BUILD )
	if( NOT LLVM_SPIRV )
		message( FATAL_ERROR "SPIR-V targets requested, but spirv-tools is not installed" )
	endif()
endif()

set( CMAKE_CLC_COMPILER ${LLVM_CLANG} )
set( CMAKE_CLC_ARCHIVE ${LLVM_LINK} )
set( CMAKE_LLAsm_PREPROCESSOR ${LLVM_CLANG} )
set( CMAKE_LLAsm_COMPILER ${LLVM_AS} )
set( CMAKE_LLAsm_ARCHIVE ${LLVM_LINK} )

# LLVM 13 enables standard includes by default
if( ${LLVM_VERSION} VERSION_GREATER "12.99.99" )
				set( CMAKE_LLAsm_FLAGS "${CMAKE_LLAsm_FLAGS} -cl-no-stdinc")
				set( CMAKE_CLC_FLAGS "${CMAKE_CLC_FLAGS} -cl-no-stdinc")
endif()

enable_language( CLC LLAsm )

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
	gfx1150 gfx1151
	gfx1200 gfx1201
)

# pkg-config file
configure_file( libclc.pc.in libclc.pc @ONLY )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/libclc.pc DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig" )
install( DIRECTORY generic/include/clc DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )

if( ENABLE_RUNTIME_SUBNORMAL )
	add_library( subnormal_use_default STATIC
		generic/libspirv/subnormal_use_default.ll )
	add_library( subnormal_disable STATIC
		generic/libspirv/subnormal_disable.ll )
	install( TARGETS subnormal_use_default subnormal_disable ARCHIVE
		DESTINATION "${CMAKE_INSTALL_DATADIR}/clc" )
endif()

find_package( Python3 REQUIRED COMPONENTS Interpreter )
file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/generic/lib/gen_convert.py clc_script_loc )
file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/generic/libspirv/gen_core_convert.py core_script_loc )
file( TO_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/generic/libspirv/gen_convert.py spirv_script_loc )

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

enable_testing()

if (LIBCLC_STANDALONE_BUILD)
  set(LIBCLC_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})
else(LIBCLC_STANDALONE_BUILD)
  set(LIBCLC_LIBRARY_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
endif(LIBCLC_STANDALONE_BUILD)
file( TO_CMAKE_PATH ${LIBCLC_LIBRARY_OUTPUT_INTDIR}/clc LIBCLC_LIBRARY_OUTPUT_INTDIR )

foreach( t ${LIBCLC_TARGETS_TO_BUILD} )
	message( "BUILDING ${t}" )
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

	#nvptx is special
	if( ${ARCH} STREQUAL nvptx OR ${ARCH} STREQUAL nvptx64 )
		set( DARCH ptx )
	else()
		set( DARCH ${ARCH} )
	endif()

	set( lib_files )
	libclc_configure_lib_source(lib_files
		LIB_DIR lib
		DIRS ${dirs} ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS}
		DEPS convert-clc.cl )
	set( libspirv_files )
	libclc_configure_lib_source(libspirv_files
		LIB_DIR libspirv
		DIRS ${dirs} ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS}
		DEPS convert-spirv.cl convert-core.cl)
	# Enumerate SOURCES* files
	set( source_list )
	foreach( l ${dirs} ${DARCH} ${DARCH}-${OS} ${DARCH}-${VENDOR}-${OS} )
		foreach( s "SOURCES" "SOURCES_${LLVM_MAJOR}.${LLVM_MINOR}" )
			file( TO_CMAKE_PATH ${l}/lib/${s} file_loc )
			file( TO_CMAKE_PATH ${CMAKE_SOURCE_DIR}/${file_loc} loc )
			# Prepend the location to give higher priority to
			# specialized implementation
			if( EXISTS ${loc} )
				set( source_list ${file_loc} ${source_list} )
			endif()
		endforeach()
	endforeach()

	# Add the generated convert.cl here to prevent adding
	# the one listed in SOURCES
	if( NOT ${ARCH} STREQUAL "spirv" AND NOT ${ARCH} STREQUAL "spirv64" )
		set( rel_files convert.cl )
		set( objects convert.cl )
		if( NOT ENABLE_RUNTIME_SUBNORMAL AND NOT ${ARCH} STREQUAL "clspv" AND
		    NOT ${ARCH} STREQUAL "clspv64" )
			list( APPEND rel_files generic/lib/subnormal_use_default.ll )
		endif()
	else()
		set( rel_files )
		set( objects )
	endif()

	foreach( l ${source_list} )
		file( READ ${l} file_list )
		string( REPLACE "\n" ";" file_list ${file_list} )
		get_filename_component( dir ${l} DIRECTORY )
		foreach( f ${file_list} )
			list( FIND objects ${f} found )
			if( found EQUAL  -1 )
				list( APPEND objects ${f} )
				list( APPEND rel_files ${dir}/${f} )
				# FIXME: This should really go away
				file( TO_CMAKE_PATH ${CMAKE_SOURCE_DIR}/${dir}/${f} src_loc )
				get_filename_component( fdir ${src_loc} DIRECTORY )

				set_source_files_properties( ${dir}/${f}
					PROPERTIES COMPILE_FLAGS "-I ${fdir}" )
			endif()
		endforeach()
	endforeach()

        # Please NOTE that variables in the foreach are not local and thus need
        # to be reset every iteration.
	foreach( d ${${t}_devices} )
		# Some targets don't have a specific GPU to target
		if( ${d} STREQUAL "none" OR ${ARCH} STREQUAL "spirv" OR ${ARCH} STREQUAL "spirv64" )
			# FIXME: Ideally we would not be tied to a specific PTX ISA version
			if( ${ARCH} STREQUAL nvptx OR ${ARCH} STREQUAL nvptx64 )
				# Disables NVVM reflection to defer to after linking
				set( flags "SHELL:-Xclang -target-feature" "SHELL:-Xclang +ptx72"
						 "SHELL:-march=sm_86" "SHELL:-mllvm --nvvm-reflect-enable=false")
			elseif( ${ARCH} STREQUAL amdgcn )
				# AMDGCN needs libclc to be compiled to high bc version since all atomic
				# clang builtins need to be accessible
				set( flags "SHELL:-mcpu=gfx940")
      elseif( ${ARCH}  STREQUAL x86_64)
        # TODO: This is used by SYCL Native Cpu, we should define an option to set this flags
        set( flags "SHELL:-Xclang -target-feature -Xclang +avx"
            "SHELL:-Xclang -fsycl-is-native-cpu"
            "SHELL:-Xclang -target-feature -Xclang +avx512f")
			else()
				set ( flags )
			endif()
			set( arch_suffix "${t}" )
		else()
			set( flags "-mcpu=${d}" )
			set( arch_suffix "${d}-${t}" )
		endif()
		message( "    DEVICE: ${d} ( ${${d}_aliases} )" )

		if ( ${ARCH} STREQUAL "spirv" OR ${ARCH} STREQUAL "spirv64" )
			if( ${ARCH} STREQUAL "spirv" )
				set( t "spir--" )
			else()
				set( t "spir64--" )
			endif()
			set( build_flags -O0 -finline-hint-functions )
			set( opt_flags -O3 )
			set( spvflags --spirv-max-version=1.1 )
		elseif( ${ARCH} STREQUAL "clspv" )
			set( t "spir--" )
			set( build_flags "-Wno-unknown-assumption")
			set( opt_flags -O3 )
		elseif( ${ARCH} STREQUAL "nvptx" OR ${ARCH} STREQUAL "nvptx64" )
			set( build_flags )
			set( opt_flags -O3 "--nvvm-reflect-enable=false" )
		elseif( ${ARCH} STREQUAL "clspv64" )
			set( t "spir64--" )
			set( build_flags "-Wno-unknown-assumption")
			set( opt_flags -O3 )
		else()
			set( build_flags )
			set( opt_flags -O3 )
		endif()

                # Enable SPIR-V builtin function declarations, so they don't
                # have to be explicity declared in the soruce.
                list( APPEND flags -Xclang -fdeclare-spirv-builtins)

                # OpenCL 3.0 extensions
                list( APPEND flags -cl-std=CL3.0 "SHELL:-Xclang" )
                string(CONCAT CL_3_0_EXTENSIONS
                  "-cl-ext="
                  "+cl_khr_fp64,"
                  "+cl_khr_fp16,"
                  "+__opencl_c_3d_image_writes,"
                  "+__opencl_c_images,"
                  "+cl_khr_3d_image_writes,"
                  "+__opencl_c_generic_address_space")
                list( APPEND flags ${CL_3_0_EXTENSIONS})

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
			TRIPLE ${t}
			TARGET_ENV libspirv
			COMPILE_OPT ${flags}
			OPT_FLAGS ${opt_flags}
			FILES ${libspirv_files}
			ALIASES ${${d}_aliases}
			GENERATE_TARGET "generate_convert_spirv.cl" "generate_convert_core.cl"
			PARENT_TARGET libspirv-builtins)

		add_libclc_builtin_set(clc-${arch_suffix}
			TRIPLE ${t}
			TARGET_ENV clc
			COMPILE_OPT ${flags}
			OPT_FLAGS ${opt_flags}
			FILES ${lib_files}
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
