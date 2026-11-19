#include "ps3_exec.h"

namespace elf {

ProcParam read_proc_param(const Image& img) {
  ProcParam p;
  const ProgramHeader* h = img.find_phdr(PT_PROC_PARAM);
  if (!h || h->filesz < 0x1C) return p;
  // sys_process_param_t: size, magic, version, sdk_version, primary_prio,
  //                      primary_stacksize, malloc_pagesize, ppc_seg, ...
  uint32_t magic = 0, sdk = 0, prio = 0, stack = 0, page = 0;
  if (!img.read32(h->vaddr + 4, magic) || magic != kProcParamMagic) return p;
  img.read32(h->vaddr + 12, sdk);
  img.read32(h->vaddr + 16, prio);
  img.read32(h->vaddr + 20, stack);
  img.read32(h->vaddr + 24, page);
  p.present = true;
  p.sdk_version = sdk;
  p.primary_prio = int32_t(prio);
  if (stack) p.primary_stacksize = stack;
  if (page) p.malloc_pagesize = page;
  return p;
}

TlsInfo read_tls(const Image& img) {
  TlsInfo t;
  if (const ProgramHeader* h = img.find_phdr(PT_TLS)) {
    t.vaddr = uint32_t(h->vaddr);
    t.filesz = uint32_t(h->filesz);
    t.memsz = uint32_t(h->memsz);
  }
  return t;
}

bool read_imports(const Image& img, std::vector<ImportModule>& out, std::string& err) {
  out.clear();
  const ProgramHeader* h = img.find_phdr(PT_PRX_PARAM);
  if (!h) return true;  // no imports
  // sys_process_prx_param: size, magic, version, unk, libent_start, libent_end,
  //                        libstub_start, libstub_end, ...
  uint32_t magic = 0, stub_start = 0, stub_end = 0;
  if (!img.read32(h->vaddr + 4, magic) || magic != kPrxParamMagic) { err = "bad PRX param magic"; return false; }
  img.read32(h->vaddr + 24, stub_start);
  img.read32(h->vaddr + 28, stub_end);
  if (stub_end < stub_start || stub_end - stub_start > 0x100000) { err = "bad libstub range"; return false; }

  // Each entry (sceLibStub / ppu_prx_module_info), 0x2C bytes:
  //  u8 size, u8 unk, u16 version, u16 attributes, u16 num_func, u16 num_var,
  //  u16 num_tlsvar, u8 hash, u8 tlshash, u8[2], u32 name, u32 nids, u32 addrs,
  //  u32 vnids, u32 vstubs, u32 unk4, u32 unk5
  for (uint32_t a = stub_start; a < stub_end;) {
    uint32_t w0 = 0;
    if (!img.read32(a, w0)) { err = "libstub entry unreadable"; return false; }
    const uint32_t size = w0 >> 24;
    if (size < 0x2C) { err = "libstub entry too small"; return false; }
    uint16_t num_func = 0, num_var = 0;
    uint32_t name = 0, nids = 0, addrs = 0, vnids = 0, vstubs = 0;
    img.read16(a + 6, num_func);
    img.read16(a + 8, num_var);
    img.read32(a + 0x10, name);
    img.read32(a + 0x14, nids);
    img.read32(a + 0x18, addrs);
    img.read32(a + 0x1C, vnids);
    img.read32(a + 0x20, vstubs);

    ImportModule m;
    if (!img.read_cstr(name, m.name, 128)) m.name = "<unreadable>";
    // addrs[] is the import table itself: the loader overwrites addrs[i] with the
    // address of the function's OPD, and the game's call stub loads it from there.
    for (uint32_t i = 0; i < num_func; ++i) {
      ImportFunc f;
      f.slot = addrs + 4 * i;
      if (!img.read32(nids + 4 * i, f.nid) || !img.read32(f.slot, f.stub)) {
        err = "import table out of bounds in " + m.name;
        return false;
      }
      m.funcs.push_back(f);
    }
    for (uint32_t i = 0; i < num_var; ++i) {
      ImportVar v;
      if (!img.read32(vnids + 4 * i, v.nid) || !img.read32(vstubs + 4 * i, v.stub)) {
        err = "import variable table out of bounds in " + m.name;
        return false;
      }
      m.vars.push_back(v);
    }
    out.push_back(std::move(m));
    a += size;
  }
  return true;
}

}  // namespace elf
