include(GNUInstallDirs)

macro(add_xpti_lib target_name)
  add_library(${target_name} STATIC ${ARGN})
  add_library(xpti::${target_name} ALIAS ${target_name})
  target_compile_definitions(${target_name} PUBLIC XPTI_STATIC_LIBRARY)
  target_include_directories(${target_name}
    PUBLIC
    $<BUILD_INTERFACE:${XPTI_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  if (MSVC)
    target_compile_options(${target_name} PRIVATE /EHsc)
  endif()

  # Set library version for the generated library.
  set_target_properties(${target_name} PROPERTIES
    VERSION ${XPTI_VERSION}
  )

  # Set the location of the library installation.
  install(TARGETS ${target_name}
    EXPORT xptiTargets
    RUNTIME DESTINATION bin COMPONENT xpti
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xpti
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xpti
  )
endmacro()

set(SOURCES
  xpti_proxy.cpp
)

add_xpti_lib(xpti ${SOURCES})

# Handle the debug version for the Microsoft compiler as a special case by
# creating a debug version of the static library that uses the flags used by
# the SYCL runtime
if (MSVC)
  add_xpti_lib(xptid ${SOURCES})
  target_compile_options(xptid PRIVATE ${XPTI_CXX_FLAGS_DEBUG})
  target_compile_options(xpti PRIVATE ${XPTI_CXX_FLAGS_RELEASE})
endif()

