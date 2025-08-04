add_subdirectory(xptitest_subscriber)

add_definitions(-DXPTI_ENABLE_INSTRUMENTATION)
add_sycl_unittest_both(XptiTraceTests XptiTraceTestsPreview OBJECT
    QueueApiFailures.cpp
    NodeCreation.cpp
    QueueIDCheck.cpp
)
target_link_libraries(XptiTraceTests PRIVATE xpti xptitest_subscriber)
target_link_libraries(XptiTraceTestsPreview PRIVATE xpti xptitest_subscriber)
