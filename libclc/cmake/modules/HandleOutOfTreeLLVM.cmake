macro(configure_out_of_tree_llvm)
  set( LIBCLC_MIN_LLVM "3.9.0" )

  if( LLVM_CONFIG )
    set (LLVM_CONFIG_FOUND 1)
    execute_process( COMMAND ${LLVM_CONFIG} "--version"
      OUTPUT_VARIABLE LLVM_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE )
    message( "LLVM version: ${LLVM_VERSION}" )

    if( ${LLVM_VERSION} VERSION_LESS ${LIBCLC_MIN_LLVM} )
      message( FATAL_ERROR "libclc needs at least LLVM ${LIBCLC_MIN_LLVM}" )
    endif()

    execute_process( COMMAND ${LLVM_CONFIG} "--libdir"
      OUTPUT_VARIABLE LLVM_LIBRARY_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE )
    execute_process( COMMAND ${LLVM_CONFIG} "--bindir"
      OUTPUT_VARIABLE LLVM_TOOLS_BINARY_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE )
    execute_process( COMMAND ${LLVM_CONFIG} "--cmakedir"
      OUTPUT_VARIABLE LLVM_CONFIG_CMAKE_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE )

    # Normalize LLVM_CMAKE_PATH. --cmakedir might contain backslashes.
    # CMake assumes slashes as PATH.
    file(TO_CMAKE_PATH ${LLVM_CONFIG_CMAKE_PATH} LLVM_CMAKE_PATH)

    # Construct LLVM version define
    string( REPLACE "." ";" LLVM_VERSION_LIST ${LLVM_VERSION} )
    list( GET LLVM_VERSION_LIST 0 LLVM_VERSION_MAJOR )
    list( GET LLVM_VERSION_LIST 1 LLVM_VERSION_MINOR )
   endif()

  if (LLVM_CMAKE_PATH AND NOT CLANG_CMAKE_PATH)
    get_filename_component(CLANG_CMAKE_PATH "${LLVM_CMAKE_PATH}" PATH)
    set(CLANG_CMAKE_PATH "${CLANG_CMAKE_PATH}/clang")
  endif()

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
  find_package(Clang REQUIRED HINTS "${CLANG_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH ${Clang_DIR})

  get_property(LLVM_CLANG TARGET clang PROPERTY LOCATION)
  get_property(LLVM_AS TARGET llvm-as PROPERTY LOCATION)
  get_property(LLVM_LINK TARGET llvm-link PROPERTY LOCATION)
  get_property(LLVM_OPT TARGET opt PROPERTY LOCATION)
  get_property(LIBCLC_REMANGLER TARGET libclc-remangler PROPERTY LOCATION)

  set(LLVM_ENABLE_PIC OFF)

  include(AddLLVM)
  include(HandleLLVMOptions)

  message("LLVM_COMPILE_FLAGS ${LLVM_COMPILE_FLAGS}")
  set(LLVM_CXX_FLAGS -I${LLVM_INCLUDE_DIR} ${CMAKE_CXX_FLAGS} ${LLVM_COMPILE_FLAGS} ${LLVM_DEFINITIONS})

  include_directories( ${LLVM_INCLUDE_DIR} ${LLVM_MAIN_INCLUDE_DIR})
endmacro(configure_out_of_tree_llvm)

configure_out_of_tree_llvm()
