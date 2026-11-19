#pragma once
// PS3 function NIDs: the first 4 bytes (little-endian) of SHA-1(name + fixed suffix).
// Games import system-library functions by NID; the name database turns NIDs back
// into readable names for logs and the --imports listing.

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace nid {

std::array<uint8_t, 20> sha1(const void* data, size_t len);
uint32_t of(std::string_view name);

// Name for a NID from the built-in database, or "" if unknown.
const std::string& name_of(uint32_t nid);

}  // namespace nid
