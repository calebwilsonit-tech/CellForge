cmake_minimum_required(VERSION 3.20)
project(Skate3Online LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CELLFORGE_BUILD_SDK_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_BINARY_DIR}/cellforge_mod_sdk")

# The mod DLL dropped into mods/
cellforge_add_mod(skate3_online
  skate3_online.cpp
  net_client.cpp
)
if(WIN32)
  target_link_libraries(skate3_online PRIVATE ws2_32)
endif()

# Standalone UDP loopback test executable
add_executable(skate3_online_test
  test_net.cpp
  net_client.cpp
)
if(WIN32)
  target_link_libraries(skate3_online_test PRIVATE ws2_32)
endif()
if(MSVC)
  target_compile_options(skate3_online_test PRIVATE /W4 /EHsc)
endif()
