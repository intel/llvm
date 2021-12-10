file(GLOB SOURCES *.cpp)

remove_definitions(-DXPTI_STATIC_LIBRARY)
add_library(xptifw SHARED ${SOURCES})
target_compile_definitions(xptifw PRIVATE -DXPTI_API_EXPORTS)

target_include_directories(xptifw PUBLIC
    $<BUILD_INTERFACE:${XPTIFW_DIR}/include>
    $<BUILD_INTERFACE:${XPTI_DIR}/include>
)

target_link_libraries(xptifw PUBLIC ${CMAKE_DL_LIBS})

if (XPTI_ENABLE_TBB)
  add_dependencies(xptifw tbb)
  target_compile_definitions(xptifw PRIVATE XPTI_USE_TBB)
  target_link_libraries(xptifw PUBLIC $<BUILD_INTERFACE:tbb>)
endif()

if (SYCL_USE_LIBCXX)
    target_link_libraries(xptifw PRIVATE "-L${SYCL_LIBCXX_LIBRARY_PATH}" -Wl,-rpath,${SYCL_LIBCXX_LIBRARY_PATH} -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc)
endif()

# Set the location of the library installation
include(GNUInstallDirs)
install(TARGETS xptifw
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT xptifw
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT xptifw
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT xptifw
)
