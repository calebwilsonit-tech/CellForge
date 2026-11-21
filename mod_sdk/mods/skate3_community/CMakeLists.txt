cmake_minimum_required(VERSION 3.20)
project(Skate3Community LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CELLFORGE_BUILD_SDK_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." "${CMAKE_BINARY_DIR}/cellforge_mod_sdk")

# The mod DLL dropped into mods/
cellforge_add_mod(skate3_community
  skate3_community.cpp
  tls_client.cpp
  rpcn.cpp
  account.cpp
)
if(WIN32)
  target_link_libraries(skate3_community PRIVATE ws2_32 user32 gdi32 secur32 crypt32)
endif()

# Standalone connectivity probe for the RPCN client
add_executable(rpcn_probe rpcn_probe.cpp rpcn.cpp tls_client.cpp)
if(WIN32)
  target_link_libraries(rpcn_probe PRIVATE ws2_32 secur32)
endif()
