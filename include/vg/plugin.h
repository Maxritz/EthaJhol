#ifndef VG_PLUGIN_H
#define VG_PLUGIN_H

#include <stdint.h>
#include "gguf.h"
#include "tensor_source.h"

#ifdef _WIN32
#define VG_PLUGIN_EXPORT __declspec(dllexport)
#else
#define VG_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define VG_PLUGIN_ABI_VERSION 1u

typedef enum VG_PluginKind {
    VG_PLUGIN_MODEL = 1,
    VG_PLUGIN_QUANT = 2,
    VG_PLUGIN_STORAGE = 3,
    VG_PLUGIN_BACKEND = 4,
    VG_PLUGIN_SAMPLER = 5
} VG_PluginKind;

typedef struct VG_PluginInfo {
    uint32_t abi_version;
    uint32_t struct_size;
    VG_PluginKind kind;
    const char *name;
    const char *version;
    const char *description;
    uint64_t capability_bits;
} VG_PluginInfo;

typedef struct VG_PluginHost {
    uint32_t abi_version;
    uint32_t struct_size;
    void (*log)(int level, const char *message, void *user);
    void *user;
} VG_PluginHost;

typedef struct VG_Plugin {
    VG_PluginInfo info;
    VG_Status (*init)(const VG_PluginHost *host);
    void (*shutdown)(void);
    int (*probe)(const VG_GGUF *model);
    void *(*create)(const VG_GGUF *model, const char *options);
    void (*destroy)(void *instance);
} VG_Plugin;

typedef const VG_Plugin *(*VG_PluginEntry)(uint32_t host_abi,
                                             const VG_PluginHost *host);

/* A shared library exports exactly this symbol by convention. */
VG_PLUGIN_EXPORT const VG_Plugin *vg_plugin_entry(uint32_t host_abi,
                                                   const VG_PluginHost *host);

#ifdef __cplusplus
}
#endif
#endif
