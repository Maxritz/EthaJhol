#include "vg/vulkan_backend.h"
#include "vg/trace.h"

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

struct VG_VKBuffer { VkBuffer buffer{}; VkDeviceMemory memory{}; void *mapped{}; size_t bytes{}; int device_local{}; int coherent{}; };

/* Descriptor-set ring per pipeline. Must exceed the number of dispatches of a
 * given pipeline recorded into ONE command buffer: a whole-token dense forward
 * records ~7 matvec dispatches per layer (see vg_vk_dense_forward), so this is
 * sized for many-layer models to avoid reusing a set already bound in the same
 * command buffer (which would alias descriptor contents). */
#define VG_VK_DESC_RING 1024

struct VG_VKPipeline {
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
    VkDescriptorSetLayout set_layout{};
    uint32_t push_size{};
    uint32_t n_bindings{};
    VkDescriptorPool pool{};
    VkDescriptorSet sets[VG_VK_DESC_RING]{};
    uint32_t ring_next{};
};

struct VG_VK {
    VkInstance instance{}; VkPhysicalDevice physical{}; VkDevice device{}; VkQueue queue{}; uint32_t family{};
    VkCommandPool command_pool{};
    VkPhysicalDeviceMemoryProperties mem{}; VG_VKInfo info{}; std::string shader_dir;
    uint64_t memory_budget_bytes{}; uint64_t memory_allocated{};

    /* Reusable submission objects (COMPASS method 3.1/4.5) */
    VkCommandBuffer cb_reuse{};
    VkFence fence{};
    int has_int_dot{};

    /* Persistent device-local cache keyed by (tensor ptr, tag) (COMPASS method 2.1).
     * LRU-managed: hot weights stay resident, cold ones evicted under budget. */
    struct CacheItem { const void *key; uint32_t tag; VG_VKBuffer *buf; uint64_t bytes; uint64_t stamp; uint64_t hits; };
    std::vector<CacheItem> cache;
    uint64_t cache_total{};
    uint64_t cache_budget{};
    uint64_t cache_clock{};
    uint64_t cache_evict_tick{};
    /* Growable host-visible scratch reused across calls (COMPASS method 2.6) */
    std::vector<std::pair<size_t, VG_VKBuffer *>> scratch;
    /* Device-local activation buffers (GPU reads/writes never touch sysmem) */
    std::vector<std::pair<size_t, VG_VKBuffer *>> devscratch;

    VG_VKPipeline matvec_i8;
    VG_VKPipeline matvec_f16;
    VG_VKPipeline matvec_f32;
    VG_VKPipeline matvec_q4_0;
    VG_VKPipeline matvec_q8_0;
    VG_VKPipeline rmsnorm;
    VG_VKPipeline rope;
    VG_VKPipeline rope_adj;
    VG_VKPipeline softmax;
    VG_VKPipeline swiglu;
    VG_VKPipeline act_quant;
    VG_VKPipeline resid_add;
    VG_VKPipeline attn_scores;
};

static uint32_t mem_type(const VkPhysicalDeviceMemoryProperties &m, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < m.memoryTypeCount; ++i) if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}
static bool read_file(const std::string &p, std::vector<uint32_t> &out) {
    FILE *f = std::fopen(p.c_str(), "rb"); if (!f) return false; std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    if (n <= 0 || (n % 4) != 0) { std::fclose(f); return false; } out.resize(static_cast<size_t>(n) / 4u); bool ok = std::fread(out.data(), 1, static_cast<size_t>(n), f) == static_cast<size_t>(n); std::fclose(f); return ok;
}
static void destroy_buffer(VG_VK *v, VG_VKBuffer *b) { if (!b) return; if (b->mapped) vkUnmapMemory(v->device, b->memory); if (b->buffer) vkDestroyBuffer(v->device, b->buffer, nullptr); if (b->memory) vkFreeMemory(v->device, b->memory, nullptr); if (v) v->memory_allocated -= b->bytes; delete b; }
static VG_Status make_buffer_ex(VG_VK *v, size_t bytes, VkBufferUsageFlags usage, int device_local, VG_VKBuffer **out) {
    if (!v || !out || bytes == 0) return VG_E_INVALID; *out = nullptr;
    if (v->memory_allocated + bytes > v->memory_budget_bytes) return VG_E_NOMEM;
    VG_VKBuffer *b = new VG_VKBuffer(); b->bytes = bytes; b->device_local = device_local;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = bytes; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(v->device, &ci, nullptr, &b->buffer) != VK_SUCCESS) { delete b; return VG_E_NOMEM; }
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(v->device, b->buffer, &req);
    uint32_t ti = UINT32_MAX;
    if (device_local) {
        /* Force PURE device-local VRAM: the AMD proprietary driver advertises
         * host-visible VIDMEM (BAR window); GPU access to it is pathologically
         * slow without full REBAR. Mirrors llama.cpp disable_host_visible_vidmem
         * (RDNA2/3/4). Prefer DEVICE_LOCAL and not HOST_VISIBLE, then any DEVICE_LOCAL. */
        for (uint32_t i = 0; i < v->mem.memoryTypeCount && ti == UINT32_MAX; ++i) {
            if (!(req.memoryTypeBits & (1u << i))) continue;
            VkMemoryPropertyFlags f = v->mem.memoryTypes[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ti = i;
        }
        if (ti == UINT32_MAX) ti = mem_type(v->mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    if (ti == UINT32_MAX) {
        /* Host-visible allocation: prefer a NON-device-local heap (system RAM).
         * The AMD proprietary driver advertises host-visible VRAM (BAR window);
         * GPU access to it is extremely slow without full REBAR. This mirrors
         * llama.cpp's RDNA2/3/4 `disable_host_visible_vidmem` fix (5.4 -> 90 t/s). */
        for (int pass = 0; pass < 4 && ti == UINT32_MAX; ++pass) {
            for (uint32_t i = 0; i < v->mem.memoryTypeCount; ++i) {
                if (!(req.memoryTypeBits & (1u << i))) continue;
                VkMemoryPropertyFlags f = v->mem.memoryTypes[i].propertyFlags;
                int hv = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
                int ch = (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                int ca = (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
                int dev = (v->mem.memoryHeaps[v->mem.memoryTypes[i].heapIndex].flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                if (!hv) continue;
                /* Prefer CPU-CACHED system RAM: reading GPU-written staging
                 * (e.g. the 128K logits) from write-combined memory is ~13x
                 * slower and dominated the per-token time. */
                if (pass == 0 && ca && ch && !dev) { ti = i; break; }
                if (pass == 1 && ch && !dev) { ti = i; break; }
                if (pass == 2 && !dev) { ti = i; break; }
                if (pass == 3 && ch) { ti = i; break; }
            }
        }
    }
    if (ti == UINT32_MAX) ti = mem_type(v->mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (ti == UINT32_MAX) { destroy_buffer(v, b); return VG_E_UNSUPPORTED; }
    b->device_local = (v->mem.memoryTypes[ti].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 1 : 0;
    b->coherent = (v->mem.memoryTypes[ti].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 1 : 0;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = ti;
    if (vkAllocateMemory(v->device, &ai, nullptr, &b->memory) != VK_SUCCESS || vkBindBufferMemory(v->device, b->buffer, b->memory, 0) != VK_SUCCESS) { destroy_buffer(v, b); return VG_E_NOMEM; }
    if (!b->device_local) { if (vkMapMemory(v->device, b->memory, 0, req.size, 0, &b->mapped) != VK_SUCCESS) { destroy_buffer(v, b); return VG_E_NOMEM; } }
    *out = b; v->memory_allocated += bytes; return VG_OK;
}
static VG_Status make_buffer(VG_VK *v, size_t bytes, VkBufferUsageFlags usage, VG_VKBuffer **out) {
    return make_buffer_ex(v, bytes, usage, 0, out);
}
/* Copy host data into a (possibly device-local) buffer via a staging buffer. */
static VG_Status submit_one(VG_VK *v, const std::function<void(VkCommandBuffer)> &record);
static VG_Status upload_buffer(VG_VK *v, VG_VKBuffer *dst, const void *data, size_t bytes) {
    if (dst->mapped) { std::memcpy(dst->mapped, data, bytes); if (!dst->coherent) { VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = dst->memory; r.size = VK_WHOLE_SIZE; vkFlushMappedMemoryRanges(v->device, 1, &r); } return VG_OK; }
    /* Persistent host-visible staging reused across uploads (COMPASS 2.6). The
     * old path allocated a staging buffer + command buffer and drained the whole
     * queue with vkQueueWaitIdle for EVERY expert miss, which dominated MoE
     * decode. Reuse the scratch buffer and the shared CB/fence instead. */
    VG_VKBuffer *staging = nullptr;
    VG_Status s = vg_vk_scratch_acquire(v, 900u, bytes, &staging);
    if (s != VG_OK) return s;
    std::memcpy(staging->mapped, data, bytes);
    if (!staging->coherent) { VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = staging->memory; r.size = VK_WHOLE_SIZE; vkFlushMappedMemoryRanges(v->device, 1, &r); }
    return submit_one(v, [&](VkCommandBuffer cb) {
        VkBufferCopy region{}; region.size = bytes;
        vkCmdCopyBuffer(cb, staging->buffer, dst->buffer, 1, &region);
    });
}
static VG_Status submit_one(VG_VK *v, const std::function<void(VkCommandBuffer)> &record) {
    VkCommandBuffer cb = v->cb_reuse;
    if (!cb) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool = v->command_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(v->device, &ai, &cb) != VK_SUCCESS) return VG_E_NOMEM;
        v->cb_reuse = cb;
    }
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VG_Status st = VG_OK;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) st = VG_E_IO; else { record(cb); if (vkEndCommandBuffer(cb) != VK_SUCCESS) st = VG_E_IO; }
    if (st == VG_OK) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        vkResetFences(v->device, 1, &v->fence);
        if (vkQueueSubmit(v->queue, 1, &si, v->fence) != VK_SUCCESS) st = VG_E_IO;
        else if (vkWaitForFences(v->device, 1, &v->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) st = VG_E_IO;
    }
    return st;
}

/* Persistent device-local cache: upload once, stay in VRAM (COMPASS 2.1). */
VG_Status vg_vk_cache_get(VG_VK *v, const void *key, uint32_t tag, VG_VKBuffer **out) {
    if (!v || !out || !key) return VG_E_INVALID;
    for (auto &e : v->cache) if (e.key == key && e.tag == tag) { ++e.hits; e.stamp = ++v->cache_clock; *out = e.buf; return VG_OK; }
    return VG_E_UNSUPPORTED;
}
VG_Status vg_vk_cache_acquire(VG_VK *v, const void *key, uint32_t tag, const void *data, size_t bytes, VG_VKBuffer **out) {
    if (!v || !out || !key || !data || !bytes) return VG_E_INVALID;
    for (auto &e : v->cache) if (e.key == key && e.tag == tag) { ++e.hits; e.stamp = ++v->cache_clock; *out = e.buf; return VG_OK; }
    /* LRU with the in-flight entry (newest stamp) never evicted, so an in-flight
     * matvec's weight buffer cannot be freed underneath it. The budget is
     * env-overridable (VG_VK_CACHE_MB / VG_VK_CACHE_PCT) so large MoE hot sets
     * can be kept resident on machines with VRAM to spare. Clamp the effective
     * limit to the TRUE remaining device memory (minus a headroom) so a large
     * MoE working set evicts an old expert instead of hitting VG_E_NOMEM in
     * make_buffer_ex and silently dropping the whole expert to the CPU path. */
    uint64_t soft = v->cache_budget ? v->cache_budget : (v->memory_budget_bytes * 85u / 100u);
    {
        uint64_t noncache = (v->memory_allocated > v->cache_total) ? (v->memory_allocated - v->cache_total) : 0;
        const uint64_t headroom = 192ull * 1024ull * 1024ull;
        uint64_t room = (v->memory_budget_bytes > noncache + headroom) ? (v->memory_budget_bytes - noncache - headroom) : 0;
        if (soft > room) soft = room;
    }
    while (!v->cache.empty() && v->cache_total + bytes > soft) {
        size_t victim = (size_t)-1; uint64_t oldest = ~0ull;
        for (size_t i = 0; i < v->cache.size(); ++i) {
            if (v->cache[i].stamp == v->cache_clock) continue;
            if (v->cache[i].stamp < oldest) { oldest = v->cache[i].stamp; victim = i; }
        }
        if (victim == (size_t)-1) break;
        v->cache_total -= v->cache[victim].bytes;
        destroy_buffer(v, v->cache[victim].buf);
        v->cache.erase(v->cache.begin() + victim);
    }
    VG_VKBuffer *b = nullptr;
    VG_Status s = make_buffer_ex(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1, &b);
    if (s != VG_OK) return s;
    s = upload_buffer(v, b, data, bytes);
    if (s != VG_OK) { destroy_buffer(v, b); return s; }
    v->cache.push_back(VG_VK::CacheItem{key, tag, b, (uint64_t)bytes, ++v->cache_clock, 1u});
    v->cache_total += bytes;
    *out = b;
    return VG_OK;
}

/* Device-local scratch (slot-indexed): activations live in VRAM; host copies
 * go through staging buffers. This is the same principle as llama.cpp's
 * disable_host_visible_vidmem fix (GPU must never touch host-visible memory). */
static VG_Status devbuf_acquire(VG_VK *v, uint32_t slot, size_t bytes, VG_VKBuffer **out) {
    if (!v || !out || !bytes) return VG_E_INVALID;
    while (v->devscratch.size() <= slot) v->devscratch.emplace_back(0, nullptr);
    if (v->devscratch[slot].second && v->devscratch[slot].first >= bytes) { *out = v->devscratch[slot].second; return VG_OK; }
    if (v->devscratch[slot].second) destroy_buffer(v, v->devscratch[slot].second);
    VG_VKBuffer *b = nullptr;
    VG_Status s = make_buffer_ex(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1, &b);
    if (s != VG_OK) { v->devscratch[slot] = {0, nullptr}; return s; }
    v->devscratch[slot] = {bytes, b};
    *out = b;
    return VG_OK;
}

/* Host-visible scratch reused across calls, keyed by slot index (each slot is
 * a distinct buffer so x/y/xd/xq can never alias within a dispatch). */
VG_Status vg_vk_scratch_acquire(VG_VK *v, uint32_t slot, size_t bytes, VG_VKBuffer **out) {
    if (!v || !out || !bytes) return VG_E_INVALID;
    while (v->scratch.size() <= slot) v->scratch.emplace_back(0, nullptr);
    if (v->scratch[slot].second && v->scratch[slot].first >= bytes) { *out = v->scratch[slot].second; return VG_OK; }
    if (v->scratch[slot].second) destroy_buffer(v, v->scratch[slot].second);
    VG_VKBuffer *b = nullptr;
    VG_Status s = make_buffer(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &b);
    if (s != VG_OK) { v->scratch[slot] = {0, nullptr}; return s; }
    v->scratch[slot] = {bytes, b};
    *out = b;
    return VG_OK;
}

/* Public wrapper around the device-local scratch pool (persistent, grow-only). */
VG_Status vg_vk_dev_scratch_acquire(VG_VK *v, uint32_t slot, size_t bytes, VG_VKBuffer **out) {
    return devbuf_acquire(v, slot, bytes, out);
}

static void destroy_pipeline(VG_VK *v, VG_VKPipeline *p) {
    if (!p) return;
    if (p->pool) vkDestroyDescriptorPool(v->device, p->pool, nullptr);
    if (p->pipeline) vkDestroyPipeline(v->device, p->pipeline, nullptr);
    if (p->layout) vkDestroyPipelineLayout(v->device, p->layout, nullptr);
    if (p->set_layout) vkDestroyDescriptorSetLayout(v->device, p->set_layout, nullptr);
    *p = {};
}

static VG_Status create_pipeline(VG_VK *v, VG_VKPipeline *out, const char *spv_name,
                                  uint32_t n_bindings, VkDescriptorType desc_type,
                                  uint32_t push_size) {
    VkDescriptorSetLayoutBinding *b = new VkDescriptorSetLayoutBinding[n_bindings];
    for (uint32_t i = 0; i < n_bindings; ++i) {
        b[i] = {}; b[i].binding = i; b[i].descriptorType = desc_type;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sci.bindingCount = n_bindings; sci.pBindings = b;
    VG_Status st = VG_OK;
    if (vkCreateDescriptorSetLayout(v->device, &sci, nullptr, &out->set_layout) != VK_SUCCESS) { delete[] b; return VG_E_NOMEM; }
    delete[] b;

    VkPushConstantRange pc{}; pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc.offset = 0; pc.size = push_size;
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 1; lci.pSetLayouts = &out->set_layout;
    lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(v->device, &lci, nullptr, &out->layout) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_NOMEM; }

    std::string path = v->shader_dir.empty() ? spv_name : v->shader_dir + "/" + spv_name;
    std::vector<uint32_t> code;
    if (!read_file(path, code)) { destroy_pipeline(v, out); return VG_E_IO; }
    VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mci.codeSize = code.size() * sizeof(uint32_t); mci.pCode = code.data();
    VkShaderModule mod{};
    if (vkCreateShaderModule(v->device, &mci, nullptr, &mod) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_IO; }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main";
    cpi.layout = out->layout;
    VkResult pr = vkCreateComputePipelines(v->device, VK_NULL_HANDLE, 1, &cpi, nullptr, &out->pipeline);
    vkDestroyShaderModule(v->device, mod, nullptr);
    if (pr != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_UNSUPPORTED; }
    out->push_size = push_size;
    out->n_bindings = n_bindings;

    /* Descriptor ring: VG_VK_DESC_RING pre-allocated sets reused batch-to-batch,
     * so no pool/set churn per dispatch (COMPASS flaws 0.4/5.4). */
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n_bindings * VG_VK_DESC_RING};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = VG_VK_DESC_RING; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(v->device, &dpi, nullptr, &out->pool) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_NOMEM; }
    VkDescriptorSetLayout layouts[VG_VK_DESC_RING];
    for (uint32_t i = 0; i < VG_VK_DESC_RING; ++i) layouts[i] = out->set_layout;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = out->pool; dai.descriptorSetCount = VG_VK_DESC_RING; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(v->device, &dai, out->sets) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_NOMEM; }
    out->ring_next = 0;
    return VG_OK;
}

static VG_Status alloc_desc_set(VG_VK *v, VkDescriptorSetLayout layout, VkDescriptorPool *pool, VkDescriptorSet *set) {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(v->device, &pi, nullptr, pool) != VK_SUCCESS) return VG_E_NOMEM;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = *pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(v->device, &dai, set) != VK_SUCCESS) { vkDestroyDescriptorPool(v->device, *pool, nullptr); *pool = VK_NULL_HANDLE; return VG_E_NOMEM; }
    return VG_OK;
}

static void bind_buffers(VG_VK *v, VkDescriptorSet set, VG_VKBuffer **bufs, uint32_t count) {
    VkWriteDescriptorSet *wr = new VkWriteDescriptorSet[count];
    VkDescriptorBufferInfo *bi = new VkDescriptorBufferInfo[count];
    for (uint32_t i = 0; i < count; ++i) {
        bi[i] = {bufs[i]->buffer, 0, bufs[i]->bytes};
        wr[i] = {}; wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set; wr[i].dstBinding = i; wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(v->device, count, wr, 0, nullptr);
    delete[] wr; delete[] bi;
}

VG_Status vg_vk_open(const VG_VKConfig *cfg, VG_VK **out) {
    if (!out) return VG_E_INVALID; *out = nullptr; VG_VK *v = new VG_VK(); if (cfg && cfg->shader_dir) v->shader_dir = cfg->shader_dir;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName = "vg-gguf"; app.applicationVersion = 1; app.pEngineName = "vg-gguf"; app.engineVersion = 1; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app; if (vkCreateInstance(&ici, nullptr, &v->instance) != VK_SUCCESS) { delete v; return VG_E_UNSUPPORTED; }
    uint32_t n = 0; vkEnumeratePhysicalDevices(v->instance, &n, nullptr); if (!n) { vg_vk_close(v); return VG_E_UNSUPPORTED; } std::vector<VkPhysicalDevice> ds(n); vkEnumeratePhysicalDevices(v->instance, &n, ds.data());
    uint32_t chosen = cfg ? cfg->device_index : UINT32_MAX; if (chosen >= n) { chosen = 0; if (cfg && cfg->prefer_discrete) for (uint32_t i = 0; i < n; ++i) { VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(ds[i], &p); if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { chosen = i; break; } } }
    v->physical = ds[chosen]; vkGetPhysicalDeviceMemoryProperties(v->physical, &v->mem); VkPhysicalDeviceProperties pp{}; vkGetPhysicalDeviceProperties(v->physical, &pp); std::snprintf(v->info.name, sizeof(v->info.name), "%s", pp.deviceName); v->info.api_version = pp.apiVersion; v->info.vendor_id = pp.vendorID; v->info.device_id = pp.deviceID; v->info.discrete = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    for (uint32_t i = 0; i < v->mem.memoryHeapCount; ++i) if (v->mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) v->info.device_local_bytes += v->mem.memoryHeaps[i].size; v->info.device_local_budget = v->info.device_local_bytes;
    v->memory_budget_bytes = (cfg && cfg->memory_budget_bytes) ? cfg->memory_budget_bytes : v->info.device_local_bytes;
    {
        /* Soft VRAM budget for the persistent weight/expert cache. Keep headroom
         * for activations/staging and avoid dragging cold experts into VRAM.
         * Overridable: VG_VK_CACHE_MB (e.g. 4096 = 4 GiB), VG_VK_CACHE_PCT. */
        const char *mb = getenv("VG_VK_CACHE_MB");
        const char *pc = getenv("VG_VK_CACHE_PCT");
        uint64_t budget;
        if (mb && atoll(mb) > 0) budget = (uint64_t)atoll(mb) * 1024ull * 1024ull;
        else if (pc && atoll(pc) > 0) budget = v->memory_budget_bytes * (uint64_t)atoll(pc) / 100ull;
        else budget = v->memory_budget_bytes * 85ull / 100ull;
        if (budget > v->memory_budget_bytes * 90ull / 100ull) budget = v->memory_budget_bytes * 90ull / 100ull;
        v->cache_budget = budget;
    }
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(v->physical, &qn, nullptr); std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(v->physical, &qn, qs.data());
    /* Prefer a DEDICATED compute family (COMPUTE without GRAPHICS) so dispatches
     * run on the async-compute engine instead of the 3D/graphics engine
     * (COMPASS hardware table: family 1 = compute+transfer, family 0 = graphics). */
    uint32_t q_best = UINT32_MAX, q_fallback = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i) {
        if (!qs[i].queueCount || !(qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
        if (!(qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { q_best = i; break; }
        if (q_fallback == UINT32_MAX) q_fallback = i;
    }
    v->family = (q_best != UINT32_MAX) ? q_best : q_fallback;
    if (v->family == UINT32_MAX) { vg_vk_close(v); return VG_E_UNSUPPORTED; }
    float priority = 1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = v->family; qci.queueCount = 1; qci.pQueuePriorities = &priority; VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    /* Integer dot product (dp4a) for the Q8_0 kernels. */
    VkPhysicalDeviceShaderIntegerDotProductFeatures dfeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    dfeat.shaderIntegerDotProduct = VK_TRUE;
    const char *dn_ext = "VK_KHR_shader_integer_dot_product";
    {
        uint32_t en = 0; vkEnumerateDeviceExtensionProperties(v->physical, nullptr, &en, nullptr); std::vector<VkExtensionProperties> eps(en); vkEnumerateDeviceExtensionProperties(v->physical, nullptr, &en, eps.data());
        for (auto &e : eps) if (std::strcmp(e.extensionName, dn_ext) == 0) v->has_int_dot = 1;
    }
    if (v->has_int_dot) { dci.pNext = &dfeat; dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &dn_ext; }
    if (vkCreateDevice(v->physical, &dci, nullptr, &v->device) != VK_SUCCESS) { vg_vk_close(v); return VG_E_UNSUPPORTED; } vkGetDeviceQueue(v->device, v->family, 0, &v->queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = v->family; pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; if (vkCreateCommandPool(v->device, &pci, nullptr, &v->command_pool) != VK_SUCCESS) { vg_vk_close(v); return VG_E_NOMEM; }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(v->device, &fci, nullptr, &v->fence) != VK_SUCCESS) { vg_vk_close(v); return VG_E_NOMEM; }

    VG_Status s;
    s = create_pipeline(v, &v->matvec_i8, "matvec_i8.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->matvec_f16, "matvec_f16.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->matvec_f32, "matvec_f32.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->matvec_q4_0, "matvec_q4_0.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    if (v->has_int_dot) {
        s = create_pipeline(v, &v->matvec_q8_0, "matvec_q8_0.comp.spv", 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8);
        if (s != VG_OK) { vg_vk_close(v); return s; }
    }
    s = create_pipeline(v, &v->rmsnorm, "rmsnorm.comp.spv", 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->rope, "rope.comp.spv", 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->softmax, "softmax.comp.spv", 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->swiglu, "swiglu.comp.spv", 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4);
    s = create_pipeline(v, &v->act_quant, "activation_quant_q8.comp.spv", 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4);
    s = create_pipeline(v, &v->resid_add, "resid_add.comp.spv", 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->rope_adj, "rope_adj.comp.spv", 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->attn_scores, "attn_scores.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20);
    if (s != VG_OK) { vg_vk_close(v); return s; }

    *out = v; return VG_OK;
}

void vg_vk_close(VG_VK *v) {
    if (!v) return;
    if (v->device) vkDeviceWaitIdle(v->device);
    for (auto &e : v->cache) destroy_buffer(v, e.buf);
    v->cache.clear();
    for (auto &e : v->scratch) destroy_buffer(v, e.second);
    v->scratch.clear();
    if (v->fence) vkDestroyFence(v->device, v->fence, nullptr);
    destroy_pipeline(v, &v->matvec_i8);
    destroy_pipeline(v, &v->matvec_f16);
    destroy_pipeline(v, &v->matvec_f32);
    destroy_pipeline(v, &v->matvec_q4_0);
    destroy_pipeline(v, &v->matvec_q8_0);
    destroy_pipeline(v, &v->rmsnorm);
    destroy_pipeline(v, &v->rope);
    destroy_pipeline(v, &v->rope_adj);
    destroy_pipeline(v, &v->softmax);
    destroy_pipeline(v, &v->swiglu);
    destroy_pipeline(v, &v->act_quant);
    destroy_pipeline(v, &v->resid_add);
    destroy_pipeline(v, &v->attn_scores);
    if (v->command_pool) vkDestroyCommandPool(v->device, v->command_pool, nullptr);
    if (v->device) vkDestroyDevice(v->device, nullptr);
    if (v->instance) vkDestroyInstance(v->instance, nullptr);
    delete v;
}

VG_Status vg_vk_info(const VG_VK *v, VG_VKInfo *out) { if (!v || !out) return VG_E_INVALID; *out = v->info; return VG_OK; }
uint64_t vg_vk_memory_used(const VG_VK *v) { return v ? v->memory_allocated : 0; }
VG_Status vg_vk_buffer_upload(VG_VK *v, const void *data, size_t bytes, VG_VKBuffer **out) { VG_Status s = make_buffer(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out); if (s != VG_OK) return s; std::memcpy((*out)->mapped, data, bytes); VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = (*out)->memory; r.size = VK_WHOLE_SIZE; vkFlushMappedMemoryRanges(v->device, 1, &r); return VG_OK; }
VG_Status vg_vk_buffer_read(VG_VK *v, const VG_VKBuffer *b, void *data, size_t bytes) { if (!v || !b || !data || bytes > b->bytes) return VG_E_INVALID; if (!b->coherent) { VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = b->memory; r.size = VK_WHOLE_SIZE; vkInvalidateMappedMemoryRanges(v->device, 1, &r); } std::memcpy(data, b->mapped, bytes); return VG_OK; }
void vg_vk_buffer_release(VG_VK *v, VG_VKBuffer *b) { destroy_buffer(v, b); }

static void quantize_row_q8_0_host(const float *x, uint32_t nb, float *xd, uint32_t *xq) {
    for (uint32_t b = 0; b < nb; ++b) {
        const float *xb = x + (size_t)b * 32u;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) { float a = std::fabs(xb[i]); if (a > amax) amax = a; }
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        xd[b] = d;
        for (int g = 0; g < 8; ++g) {
            uint32_t w = 0;
            for (int j = 0; j < 4; ++j) {
                long q = std::lround(xb[g * 4 + j] * id);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                w |= ((uint32_t)((int)q & 0xff)) << (j * 8);
            }
            xq[(size_t)b * 8u + (uint32_t)g] = w;
        }
    }
}

/* Batch core: n independent matvecs recorded into ONE command buffer, submitted
 * once with a fence (COMPASS method 3.1), barriers between dispatches (4.3),
 * descriptor ring instead of per-op allocation (5.4), persistent weights (2.1),
 * reusable scratch (2.6). q8_fast=1 selects the repacked dp4a Q8_0 path. */
static VG_Status dispatch_matvec_batch(VG_VK *v, VG_VKPipeline *pl, const VG_VKMatvecReq *reqs, uint32_t n, int q8_fast) {
    if (!v || !pl || !reqs || !n || n > VG_VK_DESC_RING) return VG_E_INVALID;
    if (!pl->pipeline) return VG_E_UNSUPPORTED;
    uint64_t t0 = vg_trace_now_ns();
    VG_VKBuffer *xb_stg[VG_VK_DESC_RING] = {0}, *yb_stg[VG_VK_DESC_RING] = {0};
    VG_VKBuffer *xd_stg[VG_VK_DESC_RING] = {0}, *xq_stg[VG_VK_DESC_RING] = {0};
    VG_VKBuffer *xb_dev[VG_VK_DESC_RING] = {0}, *yb_dev[VG_VK_DESC_RING] = {0};
    VG_VKBuffer *xd_dev[VG_VK_DESC_RING] = {0}, *xq_dev[VG_VK_DESC_RING] = {0};

    int reuse_of[VG_VK_DESC_RING];
    for (uint32_t i = 0; i < n; ++i) {
        const VG_VKMatvecReq *r = &reqs[i];
        if (!r->weights || !r->scales || !r->x || !r->y || !r->in_dim || !r->out_dim) return VG_E_INVALID;
        reuse_of[i] = -1;
        for (uint32_t j = 0; j < i; ++j)
            if (reqs[j].x == r->x && reqs[j].in_dim == r->in_dim) { reuse_of[i] = (int)j; break; }
        if (reuse_of[i] >= 0) {
            /* Same activation vector as an earlier req in this batch: share the
             * staged/quantized device copies instead of re-staging (saves a
             * memcpy+quantize per duplicate, e.g. q/k/v all reading work_embd). */
            uint32_t j = (uint32_t)reuse_of[i];
            xb_stg[i] = xb_stg[j]; xb_dev[i] = xb_dev[j];
            xd_stg[i] = xd_stg[j]; xd_dev[i] = xd_dev[j];
            xq_stg[i] = xq_stg[j]; xq_dev[i] = xq_dev[j];
        } else {
        VG_Status s = vg_vk_scratch_acquire(v, i * 6u + 0u, (size_t)r->in_dim * sizeof(float), &xb_stg[i]);
        if (s != VG_OK) return s;
        std::memcpy(xb_stg[i]->mapped, r->x, (size_t)r->in_dim * sizeof(float));
        s = devbuf_acquire(v, i * 6u + 0u, (size_t)r->in_dim * sizeof(float), &xb_dev[i]);
        if (s != VG_OK) return s;
        if (q8_fast) {
            uint32_t nb = r->in_dim / 32u;
            if (nb == 0 || r->in_dim % 32u) return VG_E_INVALID;
            s = vg_vk_scratch_acquire(v, i * 6u + 2u, (size_t)nb * sizeof(float), &xd_stg[i]);
            if (s != VG_OK) return s;
            s = vg_vk_scratch_acquire(v, i * 6u + 3u, (size_t)nb * 8u * sizeof(uint32_t), &xq_stg[i]);
            if (s != VG_OK) return s;
            s = devbuf_acquire(v, i * 6u + 2u, (size_t)nb * sizeof(float), &xd_dev[i]);
            if (s != VG_OK) return s;
            s = devbuf_acquire(v, i * 6u + 3u, (size_t)nb * 8u * sizeof(uint32_t), &xq_dev[i]);
            if (s != VG_OK) return s;
            quantize_row_q8_0_host(r->x, nb, (float *)xd_stg[i]->mapped, (uint32_t *)xq_stg[i]->mapped);
        }
        }
        VG_Status s = vg_vk_scratch_acquire(v, i * 6u + 1u, (size_t)r->out_dim * sizeof(float), &yb_stg[i]);
        if (s != VG_OK) return s;
        s = devbuf_acquire(v, i * 6u + 1u, (size_t)r->out_dim * sizeof(float), &yb_dev[i]);
        if (s != VG_OK) return s;
    }

    uint64_t t_b = vg_trace_now_ns();
    VG_Status s = submit_one(v, [&](VkCommandBuffer cb) {
        for (uint32_t i = 0; i < n; ++i) {
            const VG_VKMatvecReq *r = &reqs[i];
            VkBufferCopy c{};
            if (reuse_of[i] < 0) {
            c.size = (size_t)r->in_dim * sizeof(float);
            vkCmdCopyBuffer(cb, xb_stg[i]->buffer, xb_dev[i]->buffer, 1, &c);
            if (q8_fast) {
                uint32_t nb = r->in_dim / 32u;
                c.size = (size_t)nb * sizeof(float);
                vkCmdCopyBuffer(cb, xd_stg[i]->buffer, xd_dev[i]->buffer, 1, &c);
                c.size = (size_t)nb * 8u * sizeof(uint32_t);
                vkCmdCopyBuffer(cb, xq_stg[i]->buffer, xq_dev[i]->buffer, 1, &c);
            }
            }
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

            VkDescriptorSet set = pl->sets[pl->ring_next];
            pl->ring_next = (pl->ring_next + 1u) % VG_VK_DESC_RING;
            if (q8_fast) {
                VG_VKBuffer *bufs[5] = {(VG_VKBuffer *)r->weights, (VG_VKBuffer *)r->scales, xd_dev[i], xq_dev[i], yb_dev[i]};
                bind_buffers(v, set, bufs, 5);
                uint32_t p[2] = {r->out_dim, r->in_dim};
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &set, 0, nullptr);
                vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), p);
                vkCmdDispatch(cb, (r->out_dim + 7u) / 8u, 1, 1);
            } else {
                VG_VKBuffer *bufs[4] = {(VG_VKBuffer *)r->weights, (VG_VKBuffer *)r->scales, xb_dev[i], yb_dev[i]};
                bind_buffers(v, set, bufs, 4);
                uint32_t p[3] = {1u, r->in_dim, r->out_dim};
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &set, 0, nullptr);
                vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), p);
                vkCmdDispatch(cb, (r->out_dim + 63u) / 64u, 1, 1);
            }
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            c.size = (size_t)r->out_dim * sizeof(float);
            vkCmdCopyBuffer(cb, yb_dev[i]->buffer, yb_stg[i]->buffer, 1, &c);
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
    });

    uint64_t t_c = vg_trace_now_ns();
    if (s == VG_OK) {
        for (uint32_t i = 0; i < n; ++i)
            s = vg_vk_buffer_read(v, yb_stg[i], reqs[i].y, (size_t)reqs[i].out_dim * sizeof(float));
    }
    {
        static uint64_t pa = 0, ps = 0, pr = 0, pc = 0;
        uint64_t t_d = vg_trace_now_ns();
        pa += t_b - t0; ps += t_c - t_b; pr += t_d - t_c; ++pc;
        if ((pc % 256u) == 0u)
            fprintf(stderr, "[vg] vk: prep=%.0fus submit=%.0fus read=%.0fus n=%llu\n",
                    (double)pa / 1000.0 / (double)pc, (double)ps / 1000.0 / (double)pc,
                    (double)pr / 1000.0 / (double)pc, (unsigned long long)pc);
    }
    vg_trace_emit("vk_matvec_batch", t0, vg_trace_now_ns(), (size_t)n, 2);
    return s;
}

/* Legacy single-shot wrapper (kept for i8/f16/f32/q4_0 callers). */
static VG_Status dispatch_matvec(VG_VK *v, VG_VKPipeline *pl, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) {
    (void)rows;
    VG_VKMatvecReq r{weights, scales, x, y, input, output};
    return dispatch_matvec_batch(v, pl, &r, 1, 0);
}

VG_Status vg_vk_matvec_q8_0_batch(VG_VK *v, const VG_VKMatvecReq *reqs, uint32_t n) {
    return dispatch_matvec_batch(v, &v->matvec_q8_0, reqs, n, 1);
}
VG_Status vg_vk_matvec_f32_batch(VG_VK *v, const VG_VKMatvecReq *reqs, uint32_t n) {
    return dispatch_matvec_batch(v, &v->matvec_f32, reqs, n, 0);
}
VG_Status vg_vk_matvec_f16_batch(VG_VK *v, const VG_VKMatvecReq *reqs, uint32_t n) {
    return dispatch_matvec_batch(v, &v->matvec_f16, reqs, n, 0);
}

VG_Status vg_vk_matvec_i8(VG_VK *v, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) {
    return dispatch_matvec(v, &v->matvec_i8, weights, scales, x, y, rows, input, output);
}

VG_Status vg_vk_matvec_f16(VG_VK *v, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) {
    return dispatch_matvec(v, &v->matvec_f16, weights, scales, x, y, rows, input, output);
}

VG_Status vg_vk_matvec_f32(VG_VK *v, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) {
    return dispatch_matvec(v, &v->matvec_f32, weights, scales, x, y, rows, input, output);
}

VG_Status vg_vk_matvec_q4_0(VG_VK *v, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) {
    return dispatch_matvec(v, &v->matvec_q4_0, weights, scales, x, y, rows, input, output);
}

VG_Status vg_vk_matvec_q8_0(VG_VK *v, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) {
    (void)rows;
    VG_VKMatvecReq r{weights, scales, x, y, input, output};
    return dispatch_matvec_batch(v, &v->matvec_q8_0, &r, 1, 1);
}

VG_Status vg_vk_rmsnorm(VG_VK *v, const VG_VKBuffer *input, const VG_VKBuffer *weight, float *out, uint32_t dim, float eps) {
    if (!v || !input || !weight || !out || !dim) return VG_E_INVALID;
    uint64_t t0 = vg_trace_now_ns();
    VG_VKBuffer *ob = nullptr;
    VG_Status s = make_buffer(v, dim * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ob);
    if (s != VG_OK) return s;

    VkDescriptorPool pool{}; VkDescriptorSet set{};
    s = alloc_desc_set(v, v->rmsnorm.set_layout, &pool, &set);
    if (s != VG_OK) { destroy_buffer(v, ob); return s; }

    VG_VKBuffer *bufs[3] = {(VG_VKBuffer *)input, (VG_VKBuffer *)weight, ob};
    bind_buffers(v, set, bufs, 3);

    s = submit_one(v, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rmsnorm.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rmsnorm.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, v->rmsnorm.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &dim);
        vkCmdPushConstants(cb, v->rmsnorm.layout, VK_SHADER_STAGE_COMPUTE_BIT, 4, 4, &eps);
        vkCmdDispatch(cb, 1, 1, 1);
    });
    if (s == VG_OK) s = vg_vk_buffer_read(v, ob, out, dim * sizeof(float));
    vg_trace_emit("vk_rmsnorm", t0, vg_trace_now_ns(), dim * sizeof(float), 1);
    vkDestroyDescriptorPool(v->device, pool, nullptr);
    destroy_buffer(v, ob);
    return s;
}

VG_Status vg_vk_softmax(VG_VK *v, VG_VKBuffer *data, uint32_t dim) {
    if (!v || !data || !dim) return VG_E_INVALID;
    uint64_t t0 = vg_trace_now_ns();

    VkDescriptorPool pool{}; VkDescriptorSet set{};
    VG_Status s = alloc_desc_set(v, v->softmax.set_layout, &pool, &set);
    if (s != VG_OK) return s;

    VG_VKBuffer *bufs[1] = {data};
    bind_buffers(v, set, bufs, 1);

    s = submit_one(v, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->softmax.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->softmax.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, v->softmax.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &dim);
        vkCmdDispatch(cb, 1, 1, 1);
    });
    vg_trace_emit("vk_softmax", t0, vg_trace_now_ns(), dim * sizeof(float), 1);
    vkDestroyDescriptorPool(v->device, pool, nullptr);
    return s;
}

VG_Status vg_vk_rope(VG_VK *v, VG_VKBuffer *data, uint32_t head_dim, uint32_t n_rot, float freq_scale, float theta_base, uint32_t pos) {
    if (!v || !data || !head_dim || !n_rot) return VG_E_INVALID;
    uint64_t t0 = vg_trace_now_ns();

    VkDescriptorPool pool{}; VkDescriptorSet set{};
    VG_Status s = alloc_desc_set(v, v->rope.set_layout, &pool, &set);
    if (s != VG_OK) return s;

    VG_VKBuffer *bufs[1] = {data};
    bind_buffers(v, set, bufs, 1);

    s = submit_one(v, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rope.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rope.layout, 0, 1, &set, 0, nullptr);
        uint32_t pc[6] = {head_dim, n_rot, 0, 0, pos, head_dim / 2};
        vkCmdPushConstants(cb, v->rope.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, pc);
        vkCmdDispatch(cb, (n_rot + 63u) / 64u, 1, 1);
    });
    vg_trace_emit("vk_rope", t0, vg_trace_now_ns(), 0, 1);
    vkDestroyDescriptorPool(v->device, pool, nullptr);
    return s;
}

VG_Status vg_vk_ffn_fused_x2(VG_VK *v,
    const VG_VKBuffer *gw0, const VG_VKBuffer *gs0, const VG_VKBuffer *uw0, const VG_VKBuffer *us0, const VG_VKBuffer *dw0, const VG_VKBuffer *ds0,
    const VG_VKBuffer *gw1, const VG_VKBuffer *gs1, const VG_VKBuffer *uw1, const VG_VKBuffer *us1, const VG_VKBuffer *dw1, const VG_VKBuffer *ds1,
    const float *x, float *y0, float *y1, uint32_t D, uint32_t FF) {
    if (!v || !gw0 || !uw0 || !dw0 || !gw1 || !uw1 || !dw1 || !x || !y0 || !y1 || !D || !FF) return VG_E_INVALID;
    if ((D % 32u) || (FF % 32u)) return VG_E_UNSUPPORTED;
    if (!v->matvec_q8_0.pipeline || !v->swiglu.pipeline || !v->act_quant.pipeline) return VG_E_UNSUPPORTED;
    uint32_t nbX = D / 32u, nbM = FF / 32u;
    VG_VKBuffer *S[4] = {0}, *B[10] = {0};
    VG_Status s;
    if ((s = vg_vk_scratch_acquire(v, 300u, (size_t)nbX * 4u, &S[0])) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 301u, (size_t)nbX * 32u, &S[1])) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 302u, (size_t)D * 4u, &S[2])) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 303u, (size_t)D * 4u, &S[3])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 300u, (size_t)nbX * 4u, &B[0])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 301u, (size_t)nbX * 32u, &B[1])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 302u, (size_t)FF * 4u, &B[2])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 303u, (size_t)FF * 4u, &B[3])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 304u, (size_t)FF * 4u, &B[4])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 305u, (size_t)nbM * 4u, &B[5])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 306u, (size_t)nbM * 32u, &B[6])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 307u, (size_t)D * 4u, &B[7])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 308u, (size_t)D * 4u, &B[8])) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 309u, (size_t)D * 4u, &B[9])) != VG_OK) return s;
    quantize_row_q8_0_host(x, nbX, (float *)S[0]->mapped, (uint32_t *)S[1]->mapped);

    VkDescriptorSet DS[10]{};
    VG_VKPipeline *mv = &v->matvec_q8_0, *sw = &v->swiglu, *aq = &v->act_quant;
    VG_VKBuffer *gw2[2] = {(VG_VKBuffer *)gw0, (VG_VKBuffer *)gw1};
    VG_VKBuffer *gs2[2] = {(VG_VKBuffer *)gs0, (VG_VKBuffer *)gs1};
    VG_VKBuffer *uw2[2] = {(VG_VKBuffer *)uw0, (VG_VKBuffer *)uw1};
    VG_VKBuffer *us2[2] = {(VG_VKBuffer *)us0, (VG_VKBuffer *)us1};
    VG_VKBuffer *dw2[2] = {(VG_VKBuffer *)dw0, (VG_VKBuffer *)dw1};
    VG_VKBuffer *ds2[2] = {(VG_VKBuffer *)ds0, (VG_VKBuffer *)ds1};
    for (int e = 0; e < 2; ++e) {
        for (int j = 0; j < 3; ++j) { DS[e*3 + j] = mv->sets[mv->ring_next]; mv->ring_next = (mv->ring_next + 1u) % VG_VK_DESC_RING; }
        { VG_VKBuffer *bb[5] = {gw2[e], gs2[e], B[0], B[1], B[2]}; bind_buffers(v, DS[e*3+0], bb, 5); }
        { VG_VKBuffer *bb[5] = {uw2[e], us2[e], B[0], B[1], B[3]}; bind_buffers(v, DS[e*3+1], bb, 5); }
        { VG_VKBuffer *bb[5] = {dw2[e], ds2[e], B[5], B[6], e == 0 ? B[8] : B[9]}; bind_buffers(v, DS[e*3+2], bb, 5); }
        DS[6 + e] = sw->sets[sw->ring_next]; sw->ring_next = (sw->ring_next + 1u) % VG_VK_DESC_RING;
        { VG_VKBuffer *bb[3] = {B[2], B[3], B[4]}; bind_buffers(v, DS[6 + e], bb, 3); }
        DS[8 + e] = aq->sets[aq->ring_next]; aq->ring_next = (aq->ring_next + 1u) % VG_VK_DESC_RING;
        { VG_VKBuffer *bb[3] = {B[4], B[5], B[6]}; bind_buffers(v, DS[8 + e], bb, 3); }
    }

    s = submit_one(v, [&](VkCommandBuffer cb) {
        VkBufferCopy c{}; c.size = (size_t)nbX * 4u;
        vkCmdCopyBuffer(cb, S[0]->buffer, B[0]->buffer, 1, &c);
        c.size = (size_t)nbX * 32u; vkCmdCopyBuffer(cb, S[1]->buffer, B[1]->buffer, 1, &c);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        auto bar = [&]() { mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr); };
        auto mvec = [&](const VkDescriptorSet &set, uint32_t rows, uint32_t cols) {
            uint32_t pp[2] = {rows, cols};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->matvec_q8_0.pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->matvec_q8_0.layout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cb, v->matvec_q8_0.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pp);
            vkCmdDispatch(cb, (rows + 7u) / 8u, 1, 1);
        };
        for (int e = 0; e < 2; ++e) {
            mvec(DS[e*3+0], FF, D); bar();
            mvec(DS[e*3+1], FF, D); bar();
            { uint32_t f = FF; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.pipeline);
              vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.layout, 0, 1, &DS[6 + e], 0, nullptr);
              vkCmdPushConstants(cb, v->swiglu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &f); vkCmdDispatch(cb, (FF + 255u) / 256u, 1, 1); } bar();
            { uint32_t nb = nbM; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.pipeline);
              vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.layout, 0, 1, &DS[8 + e], 0, nullptr);
              vkCmdPushConstants(cb, v->act_quant.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nb); vkCmdDispatch(cb, nbM, 1, 1); } bar();
            mvec(DS[e*3+2], D, FF); bar();
        }
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        c.size = (size_t)D * 4u; vkCmdCopyBuffer(cb, B[8]->buffer, S[2]->buffer, 1, &c);
        c.size = (size_t)D * 4u; vkCmdCopyBuffer(cb, B[9]->buffer, S[3]->buffer, 1, &c);
    });
    if (s == VG_OK) s = vg_vk_buffer_read(v, S[2], y0, (size_t)D * sizeof(float));
    if (s == VG_OK) s = vg_vk_buffer_read(v, S[3], y1, (size_t)D * sizeof(float));
    return s;
}

VG_Status vg_vk_moe_head(VG_VK *v,
        const VG_VKBuffer *o_w, const VG_VKBuffer *o_s,
        const float *attn, const float *resid,
        const VG_VKBuffer *norm_w,
        const VG_VKBuffer *route_w, const VG_VKBuffer *route_s,
        float *hidden_out, float *logits_out, uint32_t D, uint32_t NE, float eps) {
    if (!v || !o_w || !o_s || !attn || !resid || !norm_w || !route_w || !route_s || !hidden_out || !logits_out || !D || !NE) return VG_E_INVALID;
    if (D % 32u) return VG_E_UNSUPPORTED;
    if (!v->matvec_q8_0.pipeline || !v->rmsnorm.pipeline || !v->act_quant.pipeline || !v->resid_add.pipeline) return VG_E_UNSUPPORTED;
    uint32_t nb = D / 32u;
    VG_VKBuffer *S[5] = {0}, *B[9] = {0};
    VG_Status s;
    if ((s = vg_vk_scratch_acquire(v, 400u, (size_t)nb * 4u, &S[0])) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 401u, (size_t)nb * 32u, &S[1])) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 402u, (size_t)D * 4u, &S[2])) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 403u, (size_t)D * 4u, &S[3])) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 404u, (size_t)NE * 4u, &S[4])) != VG_OK) return s;
    size_t dsz[9] = {(size_t)nb*4u, (size_t)nb*32u, (size_t)D*4u, (size_t)D*4u, (size_t)D*4u,
                     (size_t)D*4u, (size_t)nb*4u, (size_t)nb*32u, (size_t)NE*4u};
    for (uint32_t i = 0; i < 9; ++i) if ((s = devbuf_acquire(v, 400u + i, dsz[i], &B[i])) != VG_OK) return s;
    quantize_row_q8_0_host(attn, nb, (float *)S[0]->mapped, (uint32_t *)S[1]->mapped);
    std::memcpy(S[2]->mapped, resid, (size_t)D * sizeof(float));

    VkDescriptorSet DS[4]{}; VkDescriptorSet DSra{};
    VG_VKPipeline *mv = &v->matvec_q8_0, *rn = &v->rmsnorm, *aq = &v->act_quant, *ra = &v->resid_add;
    DS[0] = mv->sets[mv->ring_next]; mv->ring_next = (mv->ring_next + 1u) % VG_VK_DESC_RING;
    DS[1] = mv->sets[mv->ring_next]; mv->ring_next = (mv->ring_next + 1u) % VG_VK_DESC_RING;
    DS[2] = rn->sets[rn->ring_next]; rn->ring_next = (rn->ring_next + 1u) % VG_VK_DESC_RING;
    DS[3] = aq->sets[aq->ring_next]; aq->ring_next = (aq->ring_next + 1u) % VG_VK_DESC_RING;
    DSra  = ra->sets[ra->ring_next]; ra->ring_next = (ra->ring_next + 1u) % VG_VK_DESC_RING;
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)o_w, (VG_VKBuffer *)o_s, B[0], B[1], B[3]}; bind_buffers(v, DS[0], bb, 5); }      /* po   -> B3 */
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)route_w, (VG_VKBuffer *)route_s, B[6], B[7], B[8]}; bind_buffers(v, DS[1], bb, 5); } /* logits -> B8 */
    { VG_VKBuffer *bb[3] = {B[4], (VG_VKBuffer *)norm_w, B[5]}; bind_buffers(v, DS[2], bb, 3); }                            /* xn   -> B5 */
    { VG_VKBuffer *bb[3] = {B[5], B[6], B[7]}; bind_buffers(v, DS[3], bb, 3); }                                              /* quant(xn) -> B6,B7 */
    { VG_VKBuffer *bb[3] = {B[3], B[2], B[4]}; bind_buffers(v, DSra, bb, 3); }                                               /* h1   -> B4 */

    s = submit_one(v, [&](VkCommandBuffer cb) {
        VkBufferCopy c{};
        c.size = (size_t)nb * 4u;  vkCmdCopyBuffer(cb, S[0]->buffer, B[0]->buffer, 1, &c);
        c.size = (size_t)nb * 32u; vkCmdCopyBuffer(cb, S[1]->buffer, B[1]->buffer, 1, &c);
        c.size = (size_t)D * 4u;   vkCmdCopyBuffer(cb, S[2]->buffer, B[2]->buffer, 1, &c);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        auto bar = [&]() { mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr); };
        auto mvec = [&](const VkDescriptorSet &set, uint32_t rows, uint32_t cols) {
            uint32_t pp[2] = {rows, cols};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->matvec_q8_0.pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->matvec_q8_0.layout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cb, v->matvec_q8_0.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pp), pp);
            vkCmdDispatch(cb, (rows + 7u) / 8u, 1, 1);
        };
        mvec(DS[0], D, D); bar();                                        /* o_proj -> B3 */
        { uint32_t n = D; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->resid_add.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->resid_add.layout, 0, 1, &DSra, 0, nullptr);
          vkCmdPushConstants(cb, v->resid_add.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n); vkCmdDispatch(cb, (D + 255u) / 256u, 1, 1); } bar(); /* h1 */
        { struct { uint32_t d; float e; } pr = {D, eps}; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rmsnorm.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rmsnorm.layout, 0, 1, &DS[2], 0, nullptr);
          vkCmdPushConstants(cb, v->rmsnorm.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pr); vkCmdDispatch(cb, 1, 1, 1); } bar(); /* xn */
        { uint32_t nbb = nb; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.layout, 0, 1, &DS[3], 0, nullptr);
          vkCmdPushConstants(cb, v->act_quant.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nbb); vkCmdDispatch(cb, nb, 1, 1); } bar(); /* quant xn */
        mvec(DS[1], NE, D); bar();                                       /* router -> B8 */
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        c.size = (size_t)D * 4u;  vkCmdCopyBuffer(cb, B[4]->buffer, S[3]->buffer, 1, &c);
        c.size = (size_t)NE * 4u; vkCmdCopyBuffer(cb, B[8]->buffer, S[4]->buffer, 1, &c);
    });
    if (s == VG_OK) s = vg_vk_buffer_read(v, S[3], hidden_out, (size_t)D * sizeof(float));
    if (s == VG_OK) s = vg_vk_buffer_read(v, S[4], logits_out, (size_t)NE * sizeof(float));
    return s;
}

VG_Status vg_vk_dense_forward(VG_VK *v, const VG_VKDenseLayer *L, uint32_t n) {
    if (!v || !L || !n) return VG_E_INVALID;
    uint32_t D=L[0].D, FF=L[0].FF, Q=L[0].Q_dim, KV=L[0].KV_dim, H=L[0].H, HKV=L[0].HKV, HD=L[0].HD;
    if (!D || !FF || !Q || !KV || !H || !HKV || !HD) return VG_E_INVALID;
    if ((D%32u) || (FF%32u) || (Q%32u)) return VG_E_UNSUPPORTED;
    if (!v->matvec_q8_0.pipeline || !v->rmsnorm.pipeline || !v->act_quant.pipeline ||
        !v->swiglu.pipeline || !v->resid_add.pipeline || !v->rope_adj.pipeline || !v->attn_scores.pipeline) return VG_E_UNSUPPORTED;
    for (uint32_t i = 0; i < n; ++i)
        if (!L[i].hidden_dev || !L[i].k_cache || !L[i].v_cache ||
            L[i].D!=D || L[i].FF!=FF || L[i].Q_dim!=Q || L[i].KV_dim!=KV) return VG_E_INVALID;
    uint32_t nbD=D/32u, nbQ=Q/32u, nbFF=FF/32u;
    uint32_t kv_hdim = HKV ? (KV/HKV) : HD;

    /* Device-local activation scratch (slots 600..619), shared by all layers. */
    VG_VKBuffer *B[20] = {0};
    size_t sz[20] = {
        (size_t)D*4u, (size_t)nbD*4u, (size_t)nbD*32u,          /* 0 xn, 1 xd, 2 xq */
        (size_t)Q*4u, (size_t)KV*4u, (size_t)KV*4u,             /* 3 q, 4 k, 5 v */
        (size_t)Q*4u, (size_t)nbQ*4u, (size_t)nbQ*32u,          /* 6 att, 7 ad, 8 aq */
        (size_t)D*4u, (size_t)D*4u, (size_t)D*4u,               /* 9 po, 10 h1, 11 xn2 */
        (size_t)nbD*4u, (size_t)nbD*32u,                        /* 12 xd2, 13 xq2 */
        (size_t)FF*4u, (size_t)FF*4u, (size_t)FF*4u,            /* 14 g, 15 u, 16 mid */
        (size_t)nbFF*4u, (size_t)nbFF*32u,                      /* 17 md, 18 mq */
        (size_t)D*4u                                            /* 19 dn */
    };
    VG_Status s;
    for (uint32_t i = 0; i < 20; ++i) if ((s = devbuf_acquire(v, 600u + i, sz[i], &B[i])) != VG_OK) return s;
    VG_VKBuffer *S_h = nullptr, *S_out = nullptr;
    if (L[0].first && L[0].hidden_in) {
        if ((s = vg_vk_scratch_acquire(v, 600u, (size_t)D*4u, &S_h)) != VG_OK) return s;
        std::memcpy(S_h->mapped, L[0].hidden_in, (size_t)D*4u);
    }
    if (L[n-1].last && L[n-1].hidden_out) {
        if ((s = vg_vk_scratch_acquire(v, 601u, (size_t)D*4u, &S_out)) != VG_OK) return s;
    }

    VG_VKPipeline *mv=&v->matvec_q8_0, *rn=&v->rmsnorm, *aq=&v->act_quant, *sw=&v->swiglu, *ra=&v->resid_add, *rp=&v->rope_adj;
    s = submit_one(v, [&](VkCommandBuffer cb) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        auto bar=[&](){ mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr); };
        if (S_h) {
            VkBufferCopy c{}; c.size=(size_t)D*4u; vkCmdCopyBuffer(cb,S_h->buffer,L[0].hidden_dev->buffer,1,&c);
            mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
        }
        auto mvec=[&](const VkDescriptorSet &set,uint32_t rows,uint32_t cols){ uint32_t pp[2]={rows,cols};
            vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->matvec_q8_0.pipeline);
            vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->matvec_q8_0.layout,0,1,&set,0,nullptr);
            vkCmdPushConstants(cb,v->matvec_q8_0.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,pp); vkCmdDispatch(cb,rows,1,1); };
        auto rnorm=[&](const VkDescriptorSet &set,uint32_t dim,float eps){ struct{uint32_t d;float e;} p={dim,eps};
            vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->rmsnorm.pipeline);
            vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->rmsnorm.layout,0,1,&set,0,nullptr);
            vkCmdPushConstants(cb,v->rmsnorm.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&p); vkCmdDispatch(cb,1,1,1); };
        auto quant=[&](const VkDescriptorSet &set,uint32_t nb){ uint32_t nn=nb;
            vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->act_quant.pipeline);
            vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->act_quant.layout,0,1,&set,0,nullptr);
            vkCmdPushConstants(cb,v->act_quant.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&nn); vkCmdDispatch(cb,nb,1,1); };
        auto radd=[&](const VkDescriptorSet &set,uint32_t n0){ uint32_t nn=n0;
            vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->resid_add.pipeline);
            vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->resid_add.layout,0,1,&set,0,nullptr);
            vkCmdPushConstants(cb,v->resid_add.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&nn); vkCmdDispatch(cb,(n0+255u)/256u,1,1); };
        auto rope=[&](const VkDescriptorSet &set,uint32_t hd,uint32_t nh,uint32_t nrot,uint32_t pos,float base,uint32_t hf){
            struct{uint32_t hd,nrot,nh,pos; float base; uint32_t hf;} p={hd,nrot,nh,pos,base,hf};
            vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->rope_adj.pipeline);
            vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->rope_adj.layout,0,1,&set,0,nullptr);
            vkCmdPushConstants(cb,v->rope_adj.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,24,&p);
            uint32_t total=(nrot/2u)*nh; vkCmdDispatch(cb,(total+63u)/64u,1,1); };

        for (uint32_t li = 0; li < n; ++li) {
            const VG_VKDenseLayer *d = &L[li];
            VkDescriptorSet M[7], RN[2], AQ[4], RO[2], SW[1], RA[2], AT0{};
            for (uint32_t i=0;i<7;++i){ M[i]=mv->sets[mv->ring_next]; mv->ring_next=(mv->ring_next+1u)%VG_VK_DESC_RING; }
            for (uint32_t i=0;i<2;++i){ RN[i]=rn->sets[rn->ring_next]; rn->ring_next=(rn->ring_next+1u)%VG_VK_DESC_RING; }
            for (uint32_t i=0;i<4;++i){ AQ[i]=aq->sets[aq->ring_next]; aq->ring_next=(aq->ring_next+1u)%VG_VK_DESC_RING; }
            for (uint32_t i=0;i<2;++i){ RO[i]=rp->sets[rp->ring_next]; rp->ring_next=(rp->ring_next+1u)%VG_VK_DESC_RING; }
            SW[0]=sw->sets[sw->ring_next]; sw->ring_next=(sw->ring_next+1u)%VG_VK_DESC_RING;
            for (uint32_t i=0;i<2;++i){ RA[i]=ra->sets[ra->ring_next]; ra->ring_next=(ra->ring_next+1u)%VG_VK_DESC_RING; }
            AT0=v->attn_scores.sets[v->attn_scores.ring_next]; v->attn_scores.ring_next=(v->attn_scores.ring_next+1u)%VG_VK_DESC_RING;
            VG_VKBuffer *freq = (VG_VKBuffer *)d->rope_freqs;
            { VG_VKBuffer *bb[5]={(VG_VKBuffer*)d->q_w,(VG_VKBuffer*)d->q_s,B[1],B[2],B[3]}; bind_buffers(v,M[0],bb,5); }
            { VG_VKBuffer *bb[5]={(VG_VKBuffer*)d->k_w,(VG_VKBuffer*)d->k_s,B[1],B[2],B[4]}; bind_buffers(v,M[1],bb,5); }
            { VG_VKBuffer *bb[5]={(VG_VKBuffer*)d->v_w,(VG_VKBuffer*)d->v_s,B[1],B[2],B[5]}; bind_buffers(v,M[2],bb,5); }
            { VG_VKBuffer *bb[5]={(VG_VKBuffer*)d->o_w,(VG_VKBuffer*)d->o_s,B[7],B[8],B[9]}; bind_buffers(v,M[3],bb,5); }
            { VG_VKBuffer *bb[5]={(VG_VKBuffer*)d->gate_w,(VG_VKBuffer*)d->gate_s,B[12],B[13],B[14]}; bind_buffers(v,M[4],bb,5); }
            { VG_VKBuffer *bb[5]={(VG_VKBuffer*)d->up_w,(VG_VKBuffer*)d->up_s,B[12],B[13],B[15]}; bind_buffers(v,M[5],bb,5); }
            { VG_VKBuffer *bb[5]={(VG_VKBuffer*)d->down_w,(VG_VKBuffer*)d->down_s,B[17],B[18],B[19]}; bind_buffers(v,M[6],bb,5); }
            { VG_VKBuffer *bb[3]={d->hidden_dev,(VG_VKBuffer*)d->attn_norm_w,B[0]}; bind_buffers(v,RN[0],bb,3); }
            { VG_VKBuffer *bb[3]={B[10],(VG_VKBuffer*)d->ffn_norm_w,B[11]}; bind_buffers(v,RN[1],bb,3); }
            { VG_VKBuffer *bb[3]={B[0],B[1],B[2]}; bind_buffers(v,AQ[0],bb,3); }
            { VG_VKBuffer *bb[3]={B[6],B[7],B[8]}; bind_buffers(v,AQ[1],bb,3); }
            { VG_VKBuffer *bb[3]={B[11],B[12],B[13]}; bind_buffers(v,AQ[2],bb,3); }
            { VG_VKBuffer *bb[3]={B[16],B[17],B[18]}; bind_buffers(v,AQ[3],bb,3); }
            { VG_VKBuffer *bb[2]={B[3], freq ? freq : B[0]}; bind_buffers(v,RO[0],bb,2); }
            { VG_VKBuffer *bb[2]={B[4], freq ? freq : B[0]}; bind_buffers(v,RO[1],bb,2); }
            { VG_VKBuffer *bb[4]={B[3],d->k_cache,d->v_cache,B[6]}; bind_buffers(v,AT0,bb,4); }
            { VG_VKBuffer *bb[3]={B[14],B[15],B[16]}; bind_buffers(v,SW[0],bb,3); }
            { VG_VKBuffer *bb[3]={B[9],d->hidden_dev,B[10]}; bind_buffers(v,RA[0],bb,3); }
            { VG_VKBuffer *bb[3]={B[19],B[10],d->hidden_dev}; bind_buffers(v,RA[1],bb,3); }

            uint32_t nr_rot = d->n_rot; if (nr_rot > HD) nr_rot = HD;
            uint32_t k_rot = nr_rot; if (k_rot > kv_hdim) k_rot = kv_hdim;
            uint32_t hf = d->has_freqs ? 1u : 0u;

            rnorm(RN[0], D, d->eps); bar();
            quant(AQ[0], nbD); bar();
            mvec(M[0], Q, D); bar();
            mvec(M[1], KV, D); bar();
            mvec(M[2], KV, D); bar();
            rope(RO[0], HD, H, nr_rot, d->pos, d->freq_base, hf); bar();
            rope(RO[1], kv_hdim, HKV, k_rot, d->pos, d->freq_base, hf); bar();
            mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb,0,nullptr,0,nullptr);
            { VkBufferCopy c{}; c.size=(size_t)KV*4u; c.dstOffset=(size_t)d->pos*KV*4u;
              vkCmdCopyBuffer(cb,B[4]->buffer,d->k_cache->buffer,1,&c);
              vkCmdCopyBuffer(cb,B[5]->buffer,d->v_cache->buffer,1,&c); }
            mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
            { uint32_t p[5]={HD,H,HKV,KV,d->pos+1u};
              vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->attn_scores.pipeline);
              vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->attn_scores.layout,0,1,&AT0,0,nullptr);
              vkCmdPushConstants(cb,v->attn_scores.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,p); vkCmdDispatch(cb,H,1,1); } bar();
            quant(AQ[1], nbQ); bar();
            mvec(M[3], D, Q); bar();
            radd(RA[0], D); bar();
            rnorm(RN[1], D, d->eps); bar();
            quant(AQ[2], nbD); bar();
            mvec(M[4], FF, D); bar();
            mvec(M[5], FF, D); bar();
            { uint32_t f=FF; vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->swiglu.pipeline);
              vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,v->swiglu.layout,0,1,&SW[0],0,nullptr);
              vkCmdPushConstants(cb,v->swiglu.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&f); vkCmdDispatch(cb,(FF+255u)/256u,1,1); } bar();
            quant(AQ[3], nbFF); bar();
            mvec(M[6], D, FF); bar();
            radd(RA[1], D);
            if (li + 1u == n) {
                if (S_out) {
                    mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb,0,nullptr,0,nullptr);
                    VkBufferCopy c{}; c.size=(size_t)D*4u; vkCmdCopyBuffer(cb,d->hidden_dev->buffer,S_out->buffer,1,&c);
                }
            } else {
                bar();  /* hidden_dev is read by the next layer's attn_norm */
            }
        }
    });
    if (s == VG_OK && S_out) s = vg_vk_buffer_read(v, S_out, L[n-1].hidden_out, (size_t)D*4u);
    return s;
}

VG_Status vg_vk_dense_layer(VG_VK *v, const VG_VKDenseLayer *d) {
    return vg_vk_dense_forward(v, d, 1);
}

VG_Status vg_vk_layer_tail(VG_VK *v,
        const VG_VKBuffer *o_w, const VG_VKBuffer *o_s,
        const float *attn, const float *resid, const VG_VKBuffer *norm_w,
        const VG_VKBuffer *gw, const VG_VKBuffer *gs,
        const VG_VKBuffer *uw, const VG_VKBuffer *us,
        const VG_VKBuffer *dw2, const VG_VKBuffer *ds2,
        float *y, uint32_t D, uint32_t FF, float eps) {
    if (!v || !o_w || !o_s || !attn || !resid || !norm_w || !gw || !gs || !uw || !us || !dw2 || !ds2 || !y) return VG_E_INVALID;
    if ((D % 32u) || (FF % 32u)) return VG_E_UNSUPPORTED;
    if (!v->matvec_q8_0.pipeline || !v->rmsnorm.pipeline || !v->swiglu.pipeline || !v->act_quant.pipeline || !v->resid_add.pipeline) return VG_E_UNSUPPORTED;
    uint32_t nbO = D / 32u, nbM = FF / 32u;
    size_t scr[4] = {(size_t)nbO*4u, (size_t)nbO*32u, (size_t)D*4u, (size_t)D*4u};
    size_t dev[15] = {(size_t)nbO*4u,(size_t)nbO*32u,(size_t)D*4u,(size_t)D*4u,(size_t)D*4u,(size_t)D*4u,
                      (size_t)nbO*4u,(size_t)nbO*32u,(size_t)FF*4u,(size_t)FF*4u,(size_t)FF*4u,
                      (size_t)nbM*4u,(size_t)nbM*32u,(size_t)D*4u,(size_t)D*4u};
    VG_VKBuffer *S[4] = {0}, *B[15] = {0};
    VG_Status s;
    for (uint32_t i = 0; i < 4; ++i) if ((s = vg_vk_scratch_acquire(v, 200u + i, scr[i], &S[i])) != VG_OK) return s;
    for (uint32_t i = 0; i < 15; ++i) if ((s = devbuf_acquire(v, 200u + i, dev[i], &B[i])) != VG_OK) return s;
    quantize_row_q8_0_host(attn, nbO, (float *)S[0]->mapped, (uint32_t *)S[1]->mapped);
    memcpy(S[2]->mapped, resid, (size_t)D * sizeof(float));

    VkDescriptorSet DS[10]{};
    /* COMPASS 5.1/5.4: reuse per-pipeline descriptor-set rings instead of
     * allocating a pool+set per dispatch (was 10 pool create/destroy per layer). */
    VG_VKPipeline *mv = &v->matvec_q8_0, *rn = &v->rmsnorm, *sw = &v->swiglu, *aq = &v->act_quant, *ra = &v->resid_add;
    for (uint32_t i = 0; i < 4; ++i) { DS[i] = mv->sets[mv->ring_next]; mv->ring_next = (mv->ring_next + 1u) % VG_VK_DESC_RING; }
    DS[4] = rn->sets[rn->ring_next]; rn->ring_next = (rn->ring_next + 1u) % VG_VK_DESC_RING;
    DS[5] = sw->sets[sw->ring_next]; sw->ring_next = (sw->ring_next + 1u) % VG_VK_DESC_RING;
    for (uint32_t i = 6; i < 8; ++i) { DS[i] = aq->sets[aq->ring_next]; aq->ring_next = (aq->ring_next + 1u) % VG_VK_DESC_RING; }
    for (uint32_t i = 8; i < 10; ++i) { DS[i] = ra->sets[ra->ring_next]; ra->ring_next = (ra->ring_next + 1u) % VG_VK_DESC_RING; }
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)o_w, (VG_VKBuffer *)o_s, B[0], B[1], B[2]}; bind_buffers(v, DS[0], bb, 5); }   /* o = o_w * attn(q8) */
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)gw, (VG_VKBuffer *)gs, B[6], B[7], B[8]}; bind_buffers(v, DS[1], bb, 5); }   /* gate = gw * xn(q8) */
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)uw, (VG_VKBuffer *)us, B[6], B[7], B[9]}; bind_buffers(v, DS[2], bb, 5); }   /* up */
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)dw2, (VG_VKBuffer *)ds2, B[11], B[12], B[13]}; bind_buffers(v, DS[3], bb, 5); } /* down = dw * mid(q8) */
    { VG_VKBuffer *bb[3] = {B[4], (VG_VKBuffer *)norm_w, B[5]}; bind_buffers(v, DS[4], bb, 3); }                        /* xn = rmsnorm(h1) */
    { VG_VKBuffer *bb[3] = {B[8], B[9], B[10]}; bind_buffers(v, DS[5], bb, 3); }                                        /* mid = swiglu(g,u) */
    { VG_VKBuffer *bb[3] = {B[5], B[6], B[7]}; bind_buffers(v, DS[6], bb, 3); }                                         /* quant(xn)->B[6],B[7] */
    { VG_VKBuffer *bb[3] = {B[10], B[11], B[12]}; bind_buffers(v, DS[7], bb, 3); }                                      /* quant(mid)->B[11],B[12] */
    { VG_VKBuffer *bb[3] = {B[2], B[3], B[4]}; bind_buffers(v, DS[8], bb, 3); }                                         /* h1 = o + resid */
    { VG_VKBuffer *bb[3] = {B[13], B[4], B[14]}; bind_buffers(v, DS[9], bb, 3); }                                       /* y_dev = down + h1 */

    s = submit_one(v, [&](VkCommandBuffer cb) {
        VkBufferCopy c{};
        c.size = (size_t)nbO * 4u;  vkCmdCopyBuffer(cb, S[0]->buffer, B[0]->buffer, 1, &c);
        c.size = (size_t)nbO * 32u; vkCmdCopyBuffer(cb, S[1]->buffer, B[1]->buffer, 1, &c);
        c.size = (size_t)D * 4u;    vkCmdCopyBuffer(cb, S[2]->buffer, B[3]->buffer, 1, &c);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        auto bar = [&]() { mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr); };
        auto mvec = [&](VG_VKPipeline *pl, const VkDescriptorSet &set, uint32_t rows, uint32_t cols) {
            uint32_t pp[2] = {rows, cols};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pp);
            vkCmdDispatch(cb, (rows + 7u) / 8u, 1, 1);
        };
        mvec(&v->matvec_q8_0, DS[0], D, D);                 bar();   /* o_proj (D x D)      -> B[2] */
        { uint32_t n = D; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->resid_add.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->resid_add.layout, 0, 1, &DS[8], 0, nullptr);
          vkCmdPushConstants(cb, v->resid_add.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n); vkCmdDispatch(cb, (D + 255u) / 256u, 1, 1); } bar(); /* h1 */
        { struct { uint32_t d; float e; } pr = {D, eps}; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rmsnorm.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->rmsnorm.layout, 0, 1, &DS[4], 0, nullptr);
          vkCmdPushConstants(cb, v->rmsnorm.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &pr); vkCmdDispatch(cb, 1, 1, 1); } bar(); /* xn */
        { uint32_t nb = nbO; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.layout, 0, 1, &DS[6], 0, nullptr);
          vkCmdPushConstants(cb, v->act_quant.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nb); vkCmdDispatch(cb, nbO, 1, 1); } bar(); /* quant xn */
        mvec(&v->matvec_q8_0, DS[1], FF, D);                bar();
        mvec(&v->matvec_q8_0, DS[2], FF, D);                bar();
        { uint32_t f = FF; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.layout, 0, 1, &DS[5], 0, nullptr);
          vkCmdPushConstants(cb, v->swiglu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &f); vkCmdDispatch(cb, (FF + 255u) / 256u, 1, 1); } bar();
        { uint32_t nb = nbM; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.layout, 0, 1, &DS[7], 0, nullptr);
          vkCmdPushConstants(cb, v->act_quant.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nb); vkCmdDispatch(cb, nbM, 1, 1); } bar();
        mvec(&v->matvec_q8_0, DS[3], D, FF);                bar();
        { uint32_t n = D; vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->resid_add.pipeline);
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->resid_add.layout, 0, 1, &DS[9], 0, nullptr);
          vkCmdPushConstants(cb, v->resid_add.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n); vkCmdDispatch(cb, (D + 255u) / 256u, 1, 1); }
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        c.size = (size_t)D * 4u; vkCmdCopyBuffer(cb, B[14]->buffer, S[3]->buffer, 1, &c);
    });
    if (s == VG_OK) s = vg_vk_buffer_read(v, S[3], y, (size_t)D * sizeof(float));
    return s;
}

VG_Status vg_vk_ffn_fused(VG_VK *v,
        const VG_VKBuffer *gw, const VG_VKBuffer *gs,
        const VG_VKBuffer *uw, const VG_VKBuffer *us,
        const VG_VKBuffer *dw, const VG_VKBuffer *ds,
        const float *x, float *y, uint32_t D, uint32_t FF) {
    if (!v || !gw || !gs || !uw || !us || !dw || !ds || !x || !y || !D || !FF) return VG_E_INVALID;
    if ((D % 32u) || (FF % 32u) || !v->matvec_q8_0.pipeline || !v->swiglu.pipeline || !v->act_quant.pipeline) return VG_E_UNSUPPORTED;
    uint32_t nbX = D / 32u, nbM = FF / 32u;
    VG_VKBuffer *xd_stg=nullptr,*xq_stg=nullptr,*y_stg=nullptr;
    VG_VKBuffer *xd_dev=nullptr,*xq_dev=nullptr,*g_dev=nullptr,*u_dev=nullptr,*mid_dev=nullptr,*qd_dev=nullptr,*qq_dev=nullptr,*y_dev=nullptr;
    VG_Status s;
    if ((s = vg_vk_scratch_acquire(v, 0, (size_t)nbX * sizeof(float), &xd_stg)) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 1, (size_t)nbX * 8u * sizeof(uint32_t), &xq_stg)) != VG_OK) return s;
    if ((s = vg_vk_scratch_acquire(v, 2, (size_t)D * sizeof(float), &y_stg)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 0, (size_t)nbX * sizeof(float), &xd_dev)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 1, (size_t)nbX * 8u * sizeof(uint32_t), &xq_dev)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 2, (size_t)FF * sizeof(float), &g_dev)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 3, (size_t)FF * sizeof(float), &u_dev)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 4, (size_t)FF * sizeof(float), &mid_dev)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 5, (size_t)nbM * sizeof(float), &qd_dev)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 6, (size_t)nbM * 8u * sizeof(uint32_t), &qq_dev)) != VG_OK) return s;
    if ((s = devbuf_acquire(v, 7, (size_t)D * sizeof(float), &y_dev)) != VG_OK) return s;
    quantize_row_q8_0_host(x, nbX, (float *)xd_stg->mapped, (uint32_t *)xq_stg->mapped);

    VkDescriptorPool p1{},p2{},p3{},p4{},p5{}; VkDescriptorSet s1{},s2{},s3{},s4{},s5{};
    if ((s = alloc_desc_set(v, v->matvec_q8_0.set_layout, &p1, &s1)) != VG_OK) return s;
    if ((s = alloc_desc_set(v, v->matvec_q8_0.set_layout, &p2, &s2)) != VG_OK) return s;
    if ((s = alloc_desc_set(v, v->swiglu.set_layout, &p3, &s3)) != VG_OK) return s;
    if ((s = alloc_desc_set(v, v->act_quant.set_layout, &p4, &s4)) != VG_OK) return s;
    if ((s = alloc_desc_set(v, v->matvec_q8_0.set_layout, &p5, &s5)) != VG_OK) return s;
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)gw, (VG_VKBuffer *)gs, xd_dev, xq_dev, g_dev}; bind_buffers(v, s1, bb, 5); }
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)uw, (VG_VKBuffer *)us, xd_dev, xq_dev, u_dev}; bind_buffers(v, s2, bb, 5); }
    { VG_VKBuffer *bb[3] = {g_dev, u_dev, mid_dev}; bind_buffers(v, s3, bb, 3); }
    { VG_VKBuffer *bb[3] = {mid_dev, qd_dev, qq_dev}; bind_buffers(v, s4, bb, 3); }
    { VG_VKBuffer *bb[5] = {(VG_VKBuffer *)dw, (VG_VKBuffer *)ds, qd_dev, qq_dev, y_dev}; bind_buffers(v, s5, bb, 5); }

    s = submit_one(v, [&](VkCommandBuffer cb) {
        VkBufferCopy c{};
        c.size = (size_t)nbX * sizeof(float);
        vkCmdCopyBuffer(cb, xd_stg->buffer, xd_dev->buffer, 1, &c);
        c.size = (size_t)nbX * 8u * sizeof(uint32_t);
        vkCmdCopyBuffer(cb, xq_stg->buffer, xq_dev->buffer, 1, &c);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        auto cbar = [&]() {
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        };
        auto mvec = [&](const VkDescriptorSet &set, uint32_t rows, uint32_t cols) {
            uint32_t pp[2] = {rows, cols};
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->matvec_q8_0.pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->matvec_q8_0.layout, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cb, v->matvec_q8_0.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pp), pp);
            vkCmdDispatch(cb, (rows + 7u) / 8u, 1, 1);
        };
        mvec(s1, FF, D); cbar();
        mvec(s2, FF, D); cbar();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.layout, 0, 1, &s3, 0, nullptr);
        vkCmdPushConstants(cb, v->swiglu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &FF);
        vkCmdDispatch(cb, (FF + 255u) / 256u, 1, 1);
        cbar();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->act_quant.layout, 0, 1, &s4, 0, nullptr);
        vkCmdPushConstants(cb, v->act_quant.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nbM);
        vkCmdDispatch(cb, nbM, 1, 1);
        cbar();
        mvec(s5, D, FF);
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        c.size = (size_t)D * sizeof(float);
        vkCmdCopyBuffer(cb, y_dev->buffer, y_stg->buffer, 1, &c);
    });
    if (s == VG_OK) s = vg_vk_buffer_read(v, y_stg, y, (size_t)D * sizeof(float));
    vkDestroyDescriptorPool(v->device, p1, nullptr); vkDestroyDescriptorPool(v->device, p2, nullptr);
    vkDestroyDescriptorPool(v->device, p3, nullptr); vkDestroyDescriptorPool(v->device, p4, nullptr);
    vkDestroyDescriptorPool(v->device, p5, nullptr);
    return s;
}

VG_Status vg_vk_swiglu(VG_VK *v, const VG_VKBuffer *gate, const VG_VKBuffer *up, VG_VKBuffer *out, uint32_t dim) {
    if (!v || !gate || !up || !out || !dim) return VG_E_INVALID;
    uint64_t t0 = vg_trace_now_ns();

    VkDescriptorPool pool{}; VkDescriptorSet set{};
    VG_Status s = alloc_desc_set(v, v->swiglu.set_layout, &pool, &set);
    if (s != VG_OK) return s;

    VG_VKBuffer *bufs[3] = {(VG_VKBuffer *)gate, (VG_VKBuffer *)up, out};
    bind_buffers(v, set, bufs, 3);

    s = submit_one(v, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, v->swiglu.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cb, v->swiglu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &dim);
        vkCmdDispatch(cb, (dim + 255u) / 256u, 1, 1);
    });
    vg_trace_emit("vk_swiglu", t0, vg_trace_now_ns(), dim * sizeof(float), 1);
    vkDestroyDescriptorPool(v->device, pool, nullptr);
    return s;
}
