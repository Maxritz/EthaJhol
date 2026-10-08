#include "vg/plugin.h"
#include <stddef.h>

static const VG_PluginHost *g_host;
static VG_Status init_plugin(const VG_PluginHost *host) { g_host = host; if (g_host && g_host->log) g_host->log(1, "example plugin initialized", g_host->user); return VG_OK; }
static void shutdown_plugin(void) { g_host = NULL; }
static int probe_model(const VG_GGUF *model) { return model != NULL; }
static void *create_instance(const VG_GGUF *model, const char *options) { (void)options; return (void *)model; }
static void destroy_instance(void *instance) { (void)instance; }

static const VG_Plugin g_plugin = {
    { VG_PLUGIN_ABI_VERSION, sizeof(VG_PluginInfo), VG_PLUGIN_STORAGE, "example-storage", "0.1.0", "Minimal plugin ABI example", 0 },
    init_plugin, shutdown_plugin, probe_model, create_instance, destroy_instance
};

VG_PLUGIN_EXPORT const VG_Plugin *vg_plugin_entry(uint32_t host_abi, const VG_PluginHost *host) {
    return host_abi == VG_PLUGIN_ABI_VERSION ? &g_plugin : NULL;
}
