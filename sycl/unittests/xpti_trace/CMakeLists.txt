add_subdirectory(xptitest_subscriber)

add_definitions(-DXPTI_ENABLE_INSTRUMENTATION -gdwarf-4 -O0)
add_sycl_unittest(XptiTraceTests OBJECT
    QueueApiFailures.cpp
)

target_link_libraries(XptiTraceTests PRIVATE xptifw xpti xptitest_subscriber)
