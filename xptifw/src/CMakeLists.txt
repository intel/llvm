file(GLOB SOURCES *.cpp)

function(add_xpti_library LIB_NAME)
  remove_definitions(-DXPTI_STATIC_LIBRARY)
  add_library(${LIB_NAME} SHARED ${SOURCES})

  if (${LIB_NAME} MATCHES "xptifwd")
    set_property(TARGET ${LIB_NAME} PROPERTY
      MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
  else()
    set_property(TARGET ${LIB_NAME} PROPERTY
      MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
  endif()

  target_compile_definitions(${LIB_NAME} PRIVATE -DXPTI_API_EXPORTS)

  target_include_directories(${LIB_NAME} PUBLIC
      $<BUILD_INTERFACE:${XPTIFW_DIR}/include>
      $<BUILD_INTERFACE:${XPTI_DIR}/include>
  )

  target_link_libraries(${LIB_NAME} PUBLIC ${CMAKE_DL_LIBS})

  if (XPTI_ENABLE_TBB)
    add_dependencies(${LIB_NAME} tbb)
    target_compile_definitions(${LIB_NAME} PRIVATE XPTI_USE_TBB)
    target_link_libraries(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:tbb>)
  endif()

  if (SYCL_USE_LIBCXX)
      target_link_libraries(${LIB_NAME} PRIVATE "-L${SYCL_LIBCXX_LIBRARY_PATH}" -Wl,-rpath,${SYCL_LIBCXX_LIBRARY_PATH} -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc)
  endif()

  # Set the location of the library installation
  include(GNUInstallDirs)
  install(TARGETS ${LIB_NAME}
    RUNTIME DESTINATION bin COMPONENT xptifw
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xptifw
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT xptifw
  )
endfunction(add_xpti_library)

# MSVC provides two incompatible build variants for its CRT: release and debug
# To avoid potential issues in user code we also need to provide two kinds
# of XPTIFW for release and debug configurations.
add_xpti_library(xptifw)
if (MSVC)
  add_xpti_library(xptifwd)
endif()
