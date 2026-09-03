#ifndef VG_PLUGIN_LOADER_H
#define VG_PLUGIN_LOADER_H

#include "plugin.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VG_PluginHandle VG_PluginHandle;
VG_Status vg_plugin_load(const char *path, const VG_PluginHost *host, VG_PluginHandle **out);
const VG_Plugin *vg_plugin(const VG_PluginHandle *handle);
void vg_plugin_unload(VG_PluginHandle *handle);

#ifdef __cplusplus
}
#endif
#endif
