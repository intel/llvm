set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  CodeGen
  Core
  IPO
  AggressiveInstCombine
  InstCombine
  Instrumentation
  MC
  MCParser
  ObjCARCOpts
  Option
  ScalarOpts
  Support
  SYCLLowerIR
  TargetParser
  TransformUtils
  Vectorize
  )

# Support plugins.
if(CLANG_PLUGIN_SUPPORT)
  set(support_plugins SUPPORT_PLUGINS)
endif()

set(CLANG_BOLT_ALLOWLIST INSTRUMENT PERF LBR)
set(CLANG_BOLT OFF CACHE STRING "Apply BOLT optimization to Clang. \
May be specified as one of ${CLANG_BOLT_ALLOWLIST} to use a particular profiling \
  mechanism.")
string(TOUPPER "${CLANG_BOLT}" CLANG_BOLT)
if (CLANG_BOLT AND NOT CLANG_BOLT IN_LIST CLANG_BOLT_ALLOWLIST)
    message(FATAL_ERROR "Specified CLANG_BOLT value '${CLANG_BOLT}' is not one of ${CLANG_BOLT_ALLOWLIST}.")
endif()

if (CLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  set(CLANG_BOLT_DEPS clear-bolt-fdata llvm-bolt llvm-readobj)
  if (NOT CLANG_BOLT STREQUAL "INSTRUMENT")
    list(APPEND CLANG_BOLT_DEPS clear-perf-data)
  endif()
endif()

add_clang_tool(clang
  driver.cpp
  cc1_main.cpp
  cc1as_main.cpp
  cc1gen_reproducer_main.cpp

  DEPENDS
  intrinsics_gen
  # These generated headers are included transitively.
  target_parser_gen
  ${support_plugins}
  ${CLANG_BOLT_DEPS}
  GENERATE_DRIVER
  )

setup_host_tool(clang CLANG clang_exe clang_target)

clang_target_link_libraries(clang
  PRIVATE
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangOptions
  clangSerialization
  )

if(WIN32 AND NOT CYGWIN)
  # Prevent versioning if the buildhost is targeting for Win32.
else()
  set_target_properties(clang PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})
endif()

# Support plugins.
if(CLANG_PLUGIN_SUPPORT)
  export_executable_symbols_for_plugins(clang)
endif()

add_dependencies(clang clang-resource-headers)

if(NOT CLANG_LINKS_TO_CREATE)
  set(CLANG_LINKS_TO_CREATE clang++ clang-cl clang-cpp)
endif()

if (CLANG_ENABLE_HLSL)
  set(HLSL_LINK clang-dxc)
endif()

foreach(link ${CLANG_LINKS_TO_CREATE} ${HLSL_LINK})
  add_clang_symlink(${link} clang)
endforeach()

# Create dpclang* build-tree symlinks/copies so tests can use them without
# installing.  Use llvm_add_tool_symlink (not add_clang_symlink) to suppress
# install rules — these must NOT be installed into lib/dpcpp-N/bin/.
foreach(_link dpclang dpclang-${DPCPP_VERSION_MAJOR} dpclang++ dpclang-cl dpclang-cpp)
  llvm_add_tool_symlink(CLANG ${_link} clang ALWAYS_GENERATE)
endforeach()

# Install public dpclang* symlinks in ${CMAKE_INSTALL_BINDIR}.
# These versioned names avoid conflicts with a system LLVM installation while
# the real binaries live in ${DPCPP_INSTALL_INTERNAL_BINDIR} alongside all companion
# tools so that the driver can find them via its own executable path.
if(CLANG_BUILD_TOOLS)
  # Locate LLVM's install_symlink() helper script. LLVMInstallSymlink.cmake
  # lives in llvm/cmake/modules for an in-tree build but in
  # <llvm-install>/lib/cmake/llvm for a standalone clang build. Both candidate
  # directories are on CMAKE_MODULE_PATH, so search it (and only it) for
  # whichever one actually exists rather than hardcoding a path that only works
  # for one build layout.
  find_file(INSTALL_SYMLINK LLVMInstallSymlink.cmake
            PATHS ${CMAKE_MODULE_PATH} NO_DEFAULT_PATH)

  # Both Linux and Windows relocate the real tools into lib/dpcpp-N/bin/ so
  # they don't collide with a system LLVM. The public dpclang* entry-points go
  # into ${CMAKE_INSTALL_BINDIR}; no real binaries install there directly, so
  # make sure the directory exists.
  install(DIRECTORY DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT clang)

  # ${CMAKE_EXECUTABLE_SUFFIX} is empty on Linux and ".exe" on Windows, so the
  # same code produces dpclang / dpclang-N / ... on Linux and dpclang.exe /
  # dpclang-N.exe / ... on Windows.
  set(_exe "${CMAKE_EXECUTABLE_SUFFIX}")

  # dpclang-N points at clang++ inside the versioned tools dir; the remaining
  # names chain to dpclang-N. install_symlink resolves a relative source
  # against ${CMAKE_INSTALL_BINDIR}, so use a path relative to bin/.
  set(_abs_bindir "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
  set(_abs_toolsdir "${CMAKE_INSTALL_PREFIX}/${DPCPP_INSTALL_INTERNAL_BINDIR}")
  file(RELATIVE_PATH _dpclang_primary_target
        "${_abs_bindir}" "${_abs_toolsdir}/clang++${_exe}")

  if(LLVM_USE_SYMLINKS)
    # Linux default: relative symlinks. Resolving the symlink at startup lands
    # the driver inside the versioned tools dir where its companion tools live.
    set(_dpclang_link_or_copy create_symlink)
  else()
    # Windows default (LLVM_USE_SYMLINKS defaults to OFF on non-UNIX hosts):
    # real copies. A copied executable is not a symlink, so the driver can't
    # discover its companion tools by walking up from its own location; the
    # base ToolChain constructor seeds lib/dpcpp-N/bin/ into the program
    # search path (see clang/lib/Driver/ToolChain.cpp) to compensate.
    set(_dpclang_link_or_copy copy)
  endif()

  # dpclang-MAJOR → clang++ (in lib/dpcpp-N/bin/); dpclang/dpclang++/etc → dpclang-MAJOR
  install(SCRIPT ${INSTALL_SYMLINK}
          CODE "install_symlink(\"dpclang-${DPCPP_VERSION_MAJOR}${_exe}\" \"${_dpclang_primary_target}\" \"${CMAKE_INSTALL_BINDIR}\" \"${_dpclang_link_or_copy}\")"
          COMPONENT clang)
  install(SCRIPT ${INSTALL_SYMLINK}
          CODE "install_symlink(\"dpclang${_exe}\" \"dpclang-${DPCPP_VERSION_MAJOR}${_exe}\" \"${CMAKE_INSTALL_BINDIR}\" \"${_dpclang_link_or_copy}\")"
          COMPONENT clang)
  foreach(_link dpclang++ dpclang-cl dpclang-cpp)
    install(SCRIPT ${INSTALL_SYMLINK}
            CODE "install_symlink(\"${_link}${_exe}\" \"dpclang-${DPCPP_VERSION_MAJOR}${_exe}\" \"${CMAKE_INSTALL_BINDIR}\" \"${_dpclang_link_or_copy}\")"
            COMPONENT clang)
  endforeach()
endif()

# Configure plist creation for OS X.
set (TOOL_INFO_PLIST "Info.plist" CACHE STRING "Plist name")
if (APPLE)
  if (CLANG_VENDOR)
    set(TOOL_INFO_NAME "${CLANG_VENDOR} clang")
  else()
    set(TOOL_INFO_NAME "clang")
  endif()

  set(TOOL_INFO_UTI "${CLANG_VENDOR_UTI}")
  set(TOOL_INFO_VERSION "${CLANG_VERSION}")
  set(TOOL_INFO_BUILD_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")

  set(TOOL_INFO_PLIST_OUT "${CMAKE_CURRENT_BINARY_DIR}/${TOOL_INFO_PLIST}")

  if(LLVM_TOOL_LLVM_DRIVER_BUILD AND clang IN_LIST LLVM_DRIVER_TOOLS)
    set(TARGET_NAME llvm-driver)
  else()
    set(TARGET_NAME clang)
  endif()

  target_link_libraries(${TARGET_NAME}
    PRIVATE
    "-Wl,-sectcreate,__TEXT,__info_plist,\"${TOOL_INFO_PLIST_OUT}\"")
  configure_file("${TOOL_INFO_PLIST}.in" "${TOOL_INFO_PLIST_OUT}" @ONLY)

  set(TOOL_INFO_UTI)
  set(TOOL_INFO_NAME)
  set(TOOL_INFO_VERSION)
  set(TOOL_INFO_BUILD_VERSION)
endif()

if(CLANG_ORDER_FILE AND
    (LLVM_LINKER_IS_APPLE OR LLVM_LINKER_IS_GOLD OR LLVM_LINKER_IS_LLD))
  include(CheckLinkerFlag)

  if (LLVM_LINKER_IS_APPLE OR (LLVM_LINKER_IS_LLD AND APPLE))
    set(LINKER_ORDER_FILE_OPTION "-Wl,-order_file,${CLANG_ORDER_FILE}")
  elseif (LLVM_LINKER_IS_GOLD)
    set(LINKER_ORDER_FILE_OPTION "-Wl,--section-ordering-file,${CLANG_ORDER_FILE}")
  elseif (LLVM_LINKER_IS_LLD)
    set(LINKER_ORDER_FILE_OPTION "-Wl,--symbol-ordering-file,${CLANG_ORDER_FILE}")
  endif()

  # This is a test to ensure the actual order file works with the linker.
  check_linker_flag(CXX ${LINKER_ORDER_FILE_OPTION} LINKER_ORDER_FILE_WORKS)

  # Passing an empty order file disables some linker layout optimizations.
  # To work around this and enable workflows for re-linking when the order file
  # changes we check during configuration if the file is empty, and make it a
  # configuration dependency.
  file(READ ${CLANG_ORDER_FILE} ORDER_FILE LIMIT 20)
  if("${ORDER_FILE}" STREQUAL "\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CLANG_ORDER_FILE})
  elseif(LINKER_ORDER_FILE_WORKS)
    target_link_libraries(clang PRIVATE ${LINKER_ORDER_FILE_OPTION})
    set_target_properties(clang PROPERTIES LINK_DEPENDS ${CLANG_ORDER_FILE})
  endif()
endif()

if (CLANG_BOLT AND NOT LLVM_BUILD_INSTRUMENTED)
  # Add a clang-bolt target for backwards compatibility.
  add_custom_target(clang-bolt DEPENDS clang)

  set(CLANG_BOLT_INSTRUMENTED "clang-bolt.inst" CACHE STRING
    "Name of BOLT-instrumented Clang binary")
  set(CLANG_INSTRUMENTED ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CLANG_BOLT_INSTRUMENTED})
  set(PERF_TRAINING_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../utils/perf-training)
  set(BOLT_FDATA ${PERF_TRAINING_BINARY_DIR}/prof.fdata)
  get_llvm_lit_path(
    lit_base_dir
    lit_file_name
    ALLOW_EXTERNAL
  )
  set(LIT_COMMAND "${lit_base_dir}/${lit_file_name}")

  set(CLANG_BOLT_INPUTS $<TARGET_FILE:clang>)
  set(CLANG_INSTRUMENTED_OUTPUTS ${CLANG_INSTRUMENTED})

  # Add in dynamically linked libraries, if needs be. Currently only supported
  # on Linux because it relies on LD_PRELOAD for instrumentation.
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (CLANG_LINK_CLANG_DYLIB)
     set(CLANG_CPP_BOLT_INSTRUMENTED "clang-cxx-bolt.inst" CACHE STRING
       "Name of BOLT-instrumented Clang library")
     set(CLANG_CPP_INSTRUMENTED ${LLVM_RUNTIME_OUTPUT_INTDIR}/${CLANG_CPP_BOLT_INSTRUMENTED})
     list(APPEND CLANG_BOLT_INPUTS $<TARGET_FILE:clang-cpp>)
     list(APPEND CLANG_INSTRUMENTED_OUTPUTS ${CLANG_CPP_INSTRUMENTED})
    endif()
    if (LLVM_LINK_LLVM_DYLIB)
      set(LLVM_BOLT_INSTRUMENTED "LLVM-bolt.inst" CACHE STRING
        "Name of BOLT-instrumented LLVM library")
      set(LLVM_INSTRUMENTED ${LLVM_RUNTIME_OUTPUT_INTDIR}/${LLVM_BOLT_INSTRUMENTED})
      list(APPEND CLANG_BOLT_INPUTS $<TARGET_FILE:LLVM>)
      list(APPEND CLANG_INSTRUMENTED_OUTPUTS ${LLVM_INSTRUMENTED})
    endif()
  endif()

  # This POST_BUILD command is executed unconditionally even if the clang target
  # is already built.  We need to wrap the whole bolt optimization process in
  # a single python wrapper, so that we can first check if the binary has
  # already been optimized and then exit early with a 0 status if it has.
  add_custom_command(
    TARGET clang POST_BUILD
    COMMAND  "${Python3_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/perf-training/perf-helper.py
             bolt-optimize
             --method ${CLANG_BOLT}
             --input "${CLANG_BOLT_INPUTS}"
             --instrumented-output "${CLANG_INSTRUMENTED_OUTPUTS}"
             --fdata ${BOLT_FDATA}
             --perf-training-binary-dir ${PERF_TRAINING_BINARY_DIR}
             --readelf $<TARGET_FILE:llvm-readobj>
             --bolt $<TARGET_FILE:llvm-bolt>
             --lit "${LIT_COMMAND}"
             --merge-fdata $<TARGET_FILE:merge-fdata>
    COMMENT "Optimizing Clang with BOLT"
    USES_TERMINAL
    VERBATIM
  )
endif()
