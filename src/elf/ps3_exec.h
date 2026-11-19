#pragma once
// PS3 executable metadata: process parameters and the import (stub) tables that
// tell the loader which system-library functions the game calls.

#include <cstdint>
#include <string>
#include <vector>

#include "elf64.h"

namespace elf {

constexpr uint32_t kProcParamMagic = 0x13BCC5F6;
constexpr uint32_t kPrxParamMagic = 0x1B434CEC;

struct ProcParam {
  bool present = false;
  uint32_t sdk_version = 0;
  int32_t primary_prio = 1001;
  uint32_t primary_stacksize = 0x10000;
  uint32_t malloc_pagesize = 0x100000;
};

struct ImportFunc {
  uint32_t nid = 0;
  uint32_t slot = 0;  // guest address the loader fills with the function's OPD address
  uint32_t stub = 0;  // the slot's original contents (address of the game's call stub)
};

struct ImportVar {
  uint32_t nid = 0;
  uint32_t stub = 0;  // guest address of the variable's relocation info
};

struct ImportModule {
  std::string name;
  std::vector<ImportFunc> funcs;
  std::vector<ImportVar> vars;
};

struct TlsInfo {
  uint32_t vaddr = 0, filesz = 0, memsz = 0;
};

ProcParam read_proc_param(const Image& img);
TlsInfo read_tls(const Image& img);
// Returns false (and sets err) when the PRX param table is present but malformed.
bool read_imports(const Image& img, std::vector<ImportModule>& out, std::string& err);

}  // namespace elf
