#include "vg/plugin_loader.h"

#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
typedef HMODULE VG_Lib;
#else
#include <dlfcn.h>
typedef void *VG_Lib;
#endif

struct VG_PluginHandle { VG_Lib lib; const VG_Plugin *plugin; };
static void *sym(VG_Lib lib, const char *name) {
#ifdef _WIN32
    return (void *)(uintptr_t)GetProcAddress(lib, name);
#else
    return dlsym(lib, name);
#endif
}
VG_Status vg_plugin_load(const char *path, const VG_PluginHost *host, VG_PluginHandle **out) {
    if (!path || !out) return VG_E_INVALID; *out = NULL; VG_PluginHandle *h = (VG_PluginHandle *)calloc(1, sizeof(*h)); if (!h) return VG_E_NOMEM;
#ifdef _WIN32
    h->lib = LoadLibraryA(path);
#else
    h->lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
    if (!h->lib) { free(h); return VG_E_IO; }
    VG_PluginEntry entry = (VG_PluginEntry)sym(h->lib, "vg_plugin_entry"); if (!entry) { vg_plugin_unload(h); return VG_E_FORMAT; }
    h->plugin = entry(VG_PLUGIN_ABI_VERSION, host); if (!h->plugin || h->plugin->info.abi_version != VG_PLUGIN_ABI_VERSION || h->plugin->info.struct_size < sizeof(VG_PluginInfo)) { vg_plugin_unload(h); return VG_E_UNSUPPORTED; }
    if (h->plugin->init && h->plugin->init(host) != VG_OK) { vg_plugin_unload(h); return VG_E_BUSY; }
    *out = h; return VG_OK;
}
const VG_Plugin *vg_plugin(const VG_PluginHandle *h) { return h ? h->plugin : NULL; }
void vg_plugin_unload(VG_PluginHandle *h) {
    if (!h) return; if (h->plugin && h->plugin->shutdown) h->plugin->shutdown();
#ifdef _WIN32
    if (h->lib) FreeLibrary(h->lib);
#else
    if (h->lib) dlclose(h->lib);
#endif
    free(h);
}
