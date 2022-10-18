add_library(xpti_syclpi_collector SHARED
  syclpi_collector.cpp
)
target_link_libraries(xpti_syclpi_collector PRIVATE
  xptifw
  ${CMAKE_DL_LIBS}
)
target_include_directories(xpti_syclpi_collector PRIVATE
  ${XPTIFW_DIR}/include
  ${XPTI_DIR}/include
  ${XPTIFW_DIR}/samples/include
)

if (XPTI_ENABLE_TBB)
  target_link_libraries(xpti_syclpi_collector PRIVATE tbb)
endif()

