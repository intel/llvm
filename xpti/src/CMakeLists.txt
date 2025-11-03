include(GNUInstallDirs)

macro(add_xpti_lib target_name)
  add_library(${target_name} STATIC ${ARGN})
  target_compile_definitions(${target_name} PRIVATE -DXPTI_STATIC_LIBRARY)
  target_include_directories(${target_name} PRIVATE ${XPTI_DIR}/include)

  if (MSVC)
    target_compile_options(${target_name} PRIVATE /EHsc)
  endif()

  # Set the location of the library installation
  install(TARGETS ${target_name}
    RUNTIME DESTINATION bin COMPONENT xpti
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xpti
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xpti
  )
endmacro()

set(SOURCES
  xpti_proxy.cpp
)

add_xpti_lib(xpti ${SOURCES})
