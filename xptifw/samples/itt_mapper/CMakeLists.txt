add_library(xpti_itt_mapper SHARED
  itt_mapper.cpp
  xpti_itt_helper.cpp
)

add_library(itt_minimal SHARED itt_minimal.cpp)

target_compile_definitions(xpti_itt_mapper PRIVATE XPTI_API_EXPORTS)
target_include_directories(xpti_itt_mapper PRIVATE
  ${XPTIFW_DIR}/include
  ${XPTI_DIR}/include
  ${XPTIFW_DIR}/samples/include
  )

target_link_libraries(xpti_itt_mapper PRIVATE xptifw itt_minimal)
if(UNIX)
  target_link_libraries(xpti_itt_mapper PRIVATE dl)
endif()

if (XPTI_ENABLE_TBB)
  target_link_libraries(itt_mapper PRIVATE tbb)
endif()

# Set the location of the library installation
include(GNUInstallDirs)
install(TARGETS xpti_itt_mapper
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT xptifw
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT xptifw
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT xptifw
)
