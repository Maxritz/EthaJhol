#include "vg/vulkan_backend.h"
#include "vg/quant.h"
#include "vg/trace.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

struct VG_VKBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
    size_t bytes = 0;
};

struct VG_VKPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t push_size = 0;
    uint32_t n_bindings = 0;
    bool ready = false;
};

struct VG_VKWeight {
    VG_VKBuffer *data = nullptr;
    uint32_t ggml_type = 0;
    uint32_t out_dim = 0;
    uint32_t in_dim = 0;
};

struct VG_VK {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    /* One persistent command buffer + fence: no per-op vkQueueWaitIdle, no per-op alloc. */
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem{};
    VG_VKInfo info{};
    std::string shader_dir;
    uint64_t memory_budget_bytes = 0;
    uint64_t memory_allocated = 0;
    uint64_t uploads = 0;
    uint64_t dispatches = 0;

    VG_VKPipeline matvec_f32;
    VG_VKPipeline matvec_f16;
    VG_VKPipeline matvec_q4_0;
    VG_VKPipeline matvec_q8_0;
    VG_VKPipeline rmsnorm;
    VG_VKPipeline softmax;
    VG_VKPipeline swiglu;
    VG_VKPipeline rope;

    /* Reusable staging buffers for the vector operands. */
    VG_VKBuffer *x_buf = nullptr;
    VG_VKBuffer *y_buf = nullptr;
    size_t x_cap = 0;
    size_t y_cap = 0;
};

static const uint32_t kMaxPush = 64;

static uint32_t mem_type(const VkPhysicalDeviceMemoryProperties &m, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < m.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static bool read_file(const std::string &path, std::vector<uint32_t> &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
    long n = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
    if (n <= 0 || (n % 4) != 0) { fclose(f); return false; }
    out.resize((size_t)n / 4u);
    bool ok = fread(out.data(), 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

static VG_Status make_buffer(VG_VK *v, size_t bytes, VkBufferUsageFlags usage, VG_VKBuffer **out) {
    if (!v || !out || bytes == 0) return VG_E_INVALID;
    *out = nullptr;
    if (v->memory_allocated + bytes > v->memory_budget_bytes) return VG_E_NOMEM;
    VG_VKBuffer *b = new VG_VKBuffer();
    b->bytes = bytes;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = bytes; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(v->device, &ci, nullptr, &b->buffer) != VK_SUCCESS) { delete b; return VG_E_NOMEM; }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(v->device, b->buffer, &req);
    uint32_t ti = mem_type(v->mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (ti == UINT32_MAX) ti = mem_type(v->mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (ti == UINT32_MAX) { vkDestroyBuffer(v->device, b->buffer, nullptr); delete b; return VG_E_UNSUPPORTED; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = ti;
    if (vkAllocateMemory(v->device, &ai, nullptr, &b->memory) != VK_SUCCESS) {
        vkDestroyBuffer(v->device, b->buffer, nullptr); delete b; return VG_E_NOMEM;
    }
    if (vkBindBufferMemory(v->device, b->buffer, b->memory, 0) != VK_SUCCESS ||
        vkMapMemory(v->device, b->memory, 0, req.size, 0, &b->mapped) != VK_SUCCESS) {
        vkDestroyBuffer(v->device, b->buffer, nullptr);
        vkFreeMemory(v->device, b->memory, nullptr);
        delete b; return VG_E_NOMEM;
    }
    v->memory_allocated += bytes;
    *out = b;
    return VG_OK;
}

static void destroy_buffer(VG_VK *v, VG_VKBuffer *b) {
    if (!b) return;
    if (v && v->device) {
        if (b->mapped) vkUnmapMemory(v->device, b->memory);
        if (b->buffer) vkDestroyBuffer(v->device, b->buffer, nullptr);
        if (b->memory) vkFreeMemory(v->device, b->memory, nullptr);
    }
    if (v) v->memory_allocated -= (b->bytes <= v->memory_allocated ? b->bytes : v->memory_allocated);
    delete b;
}

static VG_Status ensure_staging(VG_VK *v, size_t x_floats, size_t y_floats) {
    if (x_floats) {
        size_t bytes = x_floats * sizeof(float);
        if (!v->x_buf || v->x_cap < bytes) {
            destroy_buffer(v, v->x_buf);
            v->x_buf = nullptr; v->x_cap = 0;
            VG_Status st = make_buffer(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v->x_buf);
            if (st != VG_OK) return st;
            v->x_cap = bytes;
        }
    }
    if (y_floats) {
        size_t bytes = y_floats * sizeof(float);
        if (!v->y_buf || v->y_cap < bytes) {
            destroy_buffer(v, v->y_buf);
            v->y_buf = nullptr; v->y_cap = 0;
            VG_Status st = make_buffer(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &v->y_buf);
            if (st != VG_OK) return st;
            v->y_cap = bytes;
        }
    }
    return VG_OK;
}

static void destroy_pipeline(VG_VK *v, VG_VKPipeline *p) {
    if (!p || !v || !v->device) return;
    if (p->pipeline) vkDestroyPipeline(v->device, p->pipeline, nullptr);
    if (p->layout) vkDestroyPipelineLayout(v->device, p->layout, nullptr);
    if (p->set_layout) vkDestroyDescriptorSetLayout(v->device, p->set_layout, nullptr);
    if (p->pool) vkDestroyDescriptorPool(v->device, p->pool, nullptr);
    *p = VG_VKPipeline{};
}

static VG_Status create_pipeline(VG_VK *v, VG_VKPipeline *out, const char *spv_name,
                                 uint32_t n_bindings, uint32_t push_size) {
    VkDescriptorSetLayoutBinding bindings[8];
    if (n_bindings > 8) return VG_E_INVALID;
    for (uint32_t i = 0; i < n_bindings; ++i) {
        bindings[i] = VkDescriptorSetLayoutBinding{};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sci.bindingCount = n_bindings; sci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(v->device, &sci, nullptr, &out->set_layout) != VK_SUCCESS) return VG_E_NOMEM;

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pc.offset = 0; pc.size = push_size;
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 1; lci.pSetLayouts = &out->set_layout;
    lci.pushConstantRangeCount = push_size ? 1u : 0u;
    lci.pPushConstantRanges = push_size ? &pc : nullptr;
    if (vkCreatePipelineLayout(v->device, &lci, nullptr, &out->layout) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_NOMEM; }

    std::string path = v->shader_dir.empty() ? spv_name : v->shader_dir + "/" + spv_name;
    std::vector<uint32_t> code;
    if (!read_file(path, code)) {
        fprintf(stderr, "[vg-vk] cannot read shader %s\n", path.c_str());
        destroy_pipeline(v, out);
        return VG_E_IO;
    }
    VkShaderModuleCreateInfo mci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mci.codeSize = code.size() * sizeof(uint32_t);
    mci.pCode = code.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(v->device, &mci, nullptr, &mod) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_IO; }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = out->layout;
    VkResult pr = vkCreateComputePipelines(v->device, VK_NULL_HANDLE, 1, &cpi, nullptr, &out->pipeline);
    vkDestroyShaderModule(v->device, mod, nullptr);
    if (pr != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_UNSUPPORTED; }

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n_bindings ? n_bindings : 1};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(v->device, &dpi, nullptr, &out->pool) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_NOMEM; }
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = out->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &out->set_layout;
    if (vkAllocateDescriptorSets(v->device, &dai, &out->set) != VK_SUCCESS) { destroy_pipeline(v, out); return VG_E_NOMEM; }

    out->push_size = push_size;
    out->n_bindings = n_bindings;
    out->ready = true;
    return VG_OK;
}

static void bind_buffers(VG_VK *v, VkDescriptorSet set, VG_VKBuffer *const *bufs, uint32_t count) {
    VkWriteDescriptorSet writes[8];
    VkDescriptorBufferInfo infos[8];
    if (count > 8) return;
    for (uint32_t i = 0; i < count; ++i) {
        infos[i].buffer = bufs[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = bufs[i]->bytes;
        writes[i] = VkWriteDescriptorSet{};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(v->device, count, writes, 0, nullptr);
}

/* Record + submit + fence wait. The command buffer and fence are reused, so a dispatch
 * costs one submit and one fence wait instead of a pool alloc, buffer alloc and
 * vkQueueWaitIdle (which also blocks every other queue). */
static VG_Status submit_recorded(VG_VK *v, const std::function<void(VkCommandBuffer)> &record) {
    vkResetCommandBuffer(v->cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(v->cmd, &bi) != VK_SUCCESS) return VG_E_IO;
    record(v->cmd);
    if (vkEndCommandBuffer(v->cmd) != VK_SUCCESS) return VG_E_IO;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &v->cmd;
    if (vkResetFences(v->device, 1, &v->fence) != VK_SUCCESS) return VG_E_IO;
    if (vkQueueSubmit(v->queue, 1, &si, v->fence) != VK_SUCCESS) return VG_E_IO;
    if (vkWaitForFences(v->device, 1, &v->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return VG_E_IO;
    ++v->dispatches;
    return VG_OK;
}

VG_Status vg_vk_open(const VG_VKConfig *cfg, VG_VK **out) {
    if (!out) return VG_E_INVALID;
    *out = nullptr;
    VG_VK *v = new VG_VK();
    if (cfg && cfg->shader_dir) v->shader_dir = cfg->shader_dir;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vg-gguf";
    app.applicationVersion = 1;
    app.pEngineName = "vg-gguf";
    app.engineVersion = 1;
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &v->instance) != VK_SUCCESS) { delete v; return VG_E_UNSUPPORTED; }

    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(v->instance, &n, nullptr) != VK_SUCCESS || n == 0) { vg_vk_close(v); return VG_E_UNSUPPORTED; }
    std::vector<VkPhysicalDevice> devices(n);
    vkEnumeratePhysicalDevices(v->instance, &n, devices.data());
    uint32_t chosen = cfg ? cfg->device_index : UINT32_MAX;
    if (chosen >= n) {
        chosen = 0;
        if (cfg && cfg->prefer_discrete) {
            for (uint32_t i = 0; i < n; ++i) {
                VkPhysicalDeviceProperties p{};
                vkGetPhysicalDeviceProperties(devices[i], &p);
                if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { chosen = i; break; }
            }
        }
    }
    v->physical = devices[chosen];
    vkGetPhysicalDeviceMemoryProperties(v->physical, &v->mem);
    VkPhysicalDeviceProperties pp{};
    vkGetPhysicalDeviceProperties(v->physical, &pp);
    snprintf(v->info.name, sizeof(v->info.name), "%s", pp.deviceName);
    v->info.api_version = pp.apiVersion;
    v->info.vendor_id = pp.vendorID;
    v->info.device_id = pp.deviceID;
    v->info.discrete = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    for (uint32_t i = 0; i < v->mem.memoryHeapCount; ++i)
        if (v->mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) v->info.device_local_bytes += v->mem.memoryHeaps[i].size;
    v->info.device_local_budget = v->info.device_local_bytes;

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(v->physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(v->physical, &qn, qs.data());
    bool found = false;
    for (uint32_t i = 0; i < qn; ++i)
        if (qs[i].queueCount && (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { v->family = i; found = true; break; }
    if (!found) { vg_vk_close(v); return VG_E_UNSUPPORTED; }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = v->family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(v->physical, &dci, nullptr, &v->device) != VK_SUCCESS) { vg_vk_close(v); return VG_E_UNSUPPORTED; }
    vkGetDeviceQueue(v->device, v->family, 0, &v->queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = v->family;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(v->device, &pci, nullptr, &v->command_pool) != VK_SUCCESS) { vg_vk_close(v); return VG_E_NOMEM; }

    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = v->command_pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(v->device, &cbi, &v->cmd) != VK_SUCCESS) { vg_vk_close(v); return VG_E_NOMEM; }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(v->device, &fci, nullptr, &v->fence) != VK_SUCCESS) { vg_vk_close(v); return VG_E_NOMEM; }

    /* Allocate a large budget for resident weights; individual uploads are still capped by
     * the real device-local size so a huge model degrades to CPU instead of failing. */
    v->memory_budget_bytes = (cfg && cfg->memory_budget_bytes) ? cfg->memory_budget_bytes : v->info.device_local_bytes;

    struct { VG_VKPipeline *p; const char *name; uint32_t bindings; uint32_t push; } specs[] = {
        {&v->matvec_f32,  "matvec_f32.comp.spv",  3, 8},
        {&v->matvec_f16,  "matvec_f16.comp.spv",  3, 8},
        {&v->matvec_q4_0, "matvec_q4_0.comp.spv", 3, 8},
        {&v->matvec_q8_0, "matvec_q8_0.comp.spv", 3, 8},
        {&v->rmsnorm,     "rmsnorm.comp.spv",     3, 8},
        {&v->softmax,     "softmax.comp.spv",     1, 4},
        {&v->swiglu,      "swiglu.comp.spv",      3, 4},
        {&v->rope,        "rope.comp.spv",        1, 24},
    };
    for (const auto &s : specs) {
        VG_Status st = create_pipeline(v, s.p, s.name, s.bindings, s.push);
        if (st != VG_OK) { vg_vk_close(v); return st; }
    }

    *out = v;
    return VG_OK;
}

void vg_vk_close(VG_VK *v) {
    if (!v) return;
    if (v->device) vkDeviceWaitIdle(v->device);
    destroy_buffer(v, v->x_buf);
    destroy_buffer(v, v->y_buf);
    destroy_pipeline(v, &v->matvec_f32);
    destroy_pipeline(v, &v->matvec_f16);
    destroy_pipeline(v, &v->matvec_q4_0);
    destroy_pipeline(v, &v->matvec_q8_0);
    destroy_pipeline(v, &v->rmsnorm);
    destroy_pipeline(v, &v->softmax);
    destroy_pipeline(v, &v->swiglu);
    destroy_pipeline(v, &v->rope);
    if (v->device) {
        if (v->fence) vkDestroyFence(v->device, v->fence, nullptr);
        if (v->command_pool) vkDestroyCommandPool(v->device, v->command_pool, nullptr);
        vkDestroyDevice(v->device, nullptr);
    }
    if (v->instance) vkDestroyInstance(v->instance, nullptr);
    delete v;
}

VG_Status vg_vk_info(const VG_VK *v, VG_VKInfo *out) {
    if (!v || !out) return VG_E_INVALID;
    *out = v->info;
    return VG_OK;
}

void vg_vk_stats(const VG_VK *v, uint64_t *uploads, uint64_t *dispatches, uint64_t *bytes_resident) {
    if (!v) return;
    if (uploads) *uploads = v->uploads;
    if (dispatches) *dispatches = v->dispatches;
    if (bytes_resident) *bytes_resident = v->memory_allocated;
}

int vg_vk_supports_type(uint32_t ggml_type) {
    switch (ggml_type) {
        case VG_QUANT_F32: return 1;
        case VG_QUANT_F16: return 1;
        case VG_QUANT_Q4_0: return 1;
        case VG_QUANT_Q8_0: return 1;
        default: return 0;
    }
}

VG_Status vg_vk_buffer_upload(VG_VK *v, const void *data, size_t bytes, VG_VKBuffer **out) {
    if (!v || !data || !bytes || !out) return VG_E_INVALID;
    VG_Status st = make_buffer(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out);
    if (st != VG_OK) return st;
    memcpy((*out)->mapped, data, bytes);
    return VG_OK;
}

VG_Status vg_vk_buffer_read(VG_VK *v, const VG_VKBuffer *b, void *data, size_t bytes) {
    if (!v || !b || !data || bytes > b->bytes) return VG_E_INVALID;
    memcpy(data, b->mapped, bytes);
    return VG_OK;
}

void vg_vk_buffer_release(VG_VK *v, VG_VKBuffer *b) { destroy_buffer(v, b); }

VG_Status vg_vk_weight_upload(VG_VK *vk, const void *packed, size_t packed_bytes,
                              uint32_t ggml_type, uint32_t out_dim, uint32_t in_dim,
                              VG_VKWeight **out) {
    if (!vk || !packed || !packed_bytes || !out) return VG_E_INVALID;
    *out = nullptr;
    if (!vg_vk_supports_type(ggml_type)) return VG_E_UNSUPPORTED;
    VG_VKWeight *w = new VG_VKWeight();
    w->ggml_type = ggml_type;
    w->out_dim = out_dim;
    w->in_dim = in_dim;
    VG_Status st = vg_vk_buffer_upload(vk, packed, packed_bytes, &w->data);
    if (st != VG_OK) { delete w; return st; }
    ++vk->uploads;
    *out = w;
    return VG_OK;
}

void vg_vk_weight_release(VG_VK *vk, VG_VKWeight *w) {
    if (!w) return;
    destroy_buffer(vk, w->data);
    delete w;
}

static VG_VKPipeline *pipeline_for(VG_VK *v, uint32_t type) {
    switch (type) {
        case VG_QUANT_F32: return &v->matvec_f32;
        case VG_QUANT_F16: return &v->matvec_f16;
        case VG_QUANT_Q4_0: return &v->matvec_q4_0;
        case VG_QUANT_Q8_0: return &v->matvec_q8_0;
        default: return nullptr;
    }
}

VG_Status vg_vk_matvec(VG_VK *vk, const VG_VKWeight *w, const float *x, float *y,
                       uint32_t out_dim, uint32_t in_dim) {
    if (!vk || !w || !x || !y || !out_dim || !in_dim) return VG_E_INVALID;
    VG_VKPipeline *pl = pipeline_for(vk, w->ggml_type);
    if (!pl || !pl->ready) return VG_E_UNSUPPORTED;
    uint64_t t0 = vg_trace_now_ns();
    VG_Status st = ensure_staging(vk, in_dim, out_dim);
    if (st != VG_OK) return st;
    memcpy(vk->x_buf->mapped, x, (size_t)in_dim * sizeof(float));

    /* matvec pipelines bind (weights, x, y). */
    VG_VKBuffer *bufs[3] = {w->data, vk->x_buf, vk->y_buf};
    bind_buffers(vk, pl->set, bufs, pl->n_bindings);
    uint32_t push[2] = {in_dim, out_dim};
    st = submit_recorded(vk, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &pl->set, 0, nullptr);
        vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
        vkCmdDispatch(cb, (out_dim + 63u) / 64u, 1, 1);
    });
    if (st == VG_OK) memcpy(y, vk->y_buf->mapped, (size_t)out_dim * sizeof(float));
    vg_trace_emit("vk_matvec", t0, vg_trace_now_ns(), (size_t)out_dim * sizeof(float), 2);
    return st;
}

static VG_Status matvec_once(VG_VK *vk, uint32_t type, const void *weights, const float *x, float *y,
                             uint32_t out_dim, uint32_t in_dim) {
    if (!vk || !weights) return VG_E_INVALID;
    const VG_QuantInfo *qi = vg_quant_info(type);
    if (!qi) return VG_E_UNSUPPORTED;
    size_t bytes = (size_t)((uint64_t)out_dim * in_dim / qi->block_size * qi->bytes_per_block);
    VG_VKWeight *w = nullptr;
    VG_Status st = vg_vk_weight_upload(vk, weights, bytes, type, out_dim, in_dim, &w);
    if (st != VG_OK) return st;
    st = vg_vk_matvec(vk, w, x, y, out_dim, in_dim);
    vg_vk_weight_release(vk, w);
    return st;
}

VG_Status vg_vk_matvec_q4_0(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    return matvec_once(vk, VG_QUANT_Q4_0, weights, x, y, out_dim, in_dim);
}
VG_Status vg_vk_matvec_q8_0(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    return matvec_once(vk, VG_QUANT_Q8_0, weights, x, y, out_dim, in_dim);
}
VG_Status vg_vk_matvec_f32(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    return matvec_once(vk, VG_QUANT_F32, weights, x, y, out_dim, in_dim);
}
VG_Status vg_vk_matvec_f16(VG_VK *vk, const void *weights, const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    return matvec_once(vk, VG_QUANT_F16, weights, x, y, out_dim, in_dim);
}

VG_Status vg_vk_rmsnorm(VG_VK *vk, const float *x, const float *weight, float *out, uint32_t dim, float eps) {
    if (!vk || !x || !weight || !out || !dim) return VG_E_INVALID;
    VG_VKPipeline *pl = &vk->rmsnorm;
    if (!pl->ready) return VG_E_UNSUPPORTED;
    VG_Status st = ensure_staging(vk, dim, dim);
    if (st != VG_OK) return st;
    VG_VKBuffer *wb = nullptr;
    st = vg_vk_buffer_upload(vk, weight, (size_t)dim * sizeof(float), &wb);
    if (st != VG_OK) return st;
    memcpy(vk->x_buf->mapped, x, (size_t)dim * sizeof(float));
    VG_VKBuffer *bufs[3] = {vk->x_buf, wb, vk->y_buf};
    bind_buffers(vk, pl->set, bufs, 3);
    uint32_t pc[2] = {dim, 0};
    memcpy(&pc[1], &eps, sizeof(float));
    st = submit_recorded(vk, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &pl->set, 0, nullptr);
        vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cb, 1, 1, 1);
    });
    if (st == VG_OK) memcpy(out, vk->y_buf->mapped, (size_t)dim * sizeof(float));
    destroy_buffer(vk, wb);
    return st;
}

VG_Status vg_vk_softmax(VG_VK *vk, float *data, uint32_t dim) {
    if (!vk || !data || !dim) return VG_E_INVALID;
    VG_VKPipeline *pl = &vk->softmax;
    if (!pl->ready) return VG_E_UNSUPPORTED;
    VG_Status st = ensure_staging(vk, dim, 0);
    if (st != VG_OK) return st;
    memcpy(vk->x_buf->mapped, data, (size_t)dim * sizeof(float));
    VG_VKBuffer *bufs[1] = {vk->x_buf};
    bind_buffers(vk, pl->set, bufs, 1);
    st = submit_recorded(vk, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &pl->set, 0, nullptr);
        vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dim), &dim);
        vkCmdDispatch(cb, 1, 1, 1);
    });
    if (st == VG_OK) memcpy(data, vk->x_buf->mapped, (size_t)dim * sizeof(float));
    return st;
}

VG_Status vg_vk_swiglu(VG_VK *vk, const float *gate, const float *up, float *out, uint32_t dim) {
    if (!vk || !gate || !up || !out || !dim) return VG_E_INVALID;
    VG_VKPipeline *pl = &vk->swiglu;
    if (!pl->ready) return VG_E_UNSUPPORTED;
    VG_Status st = ensure_staging(vk, dim, dim);
    if (st != VG_OK) return st;
    VG_VKBuffer *gb = nullptr, *ub = nullptr;
    st = vg_vk_buffer_upload(vk, gate, (size_t)dim * sizeof(float), &gb);
    if (st != VG_OK) return st;
    st = vg_vk_buffer_upload(vk, up, (size_t)dim * sizeof(float), &ub);
    if (st != VG_OK) { destroy_buffer(vk, gb); return st; }
    VG_VKBuffer *bufs[3] = {gb, ub, vk->y_buf};
    bind_buffers(vk, pl->set, bufs, 3);
    st = submit_recorded(vk, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &pl->set, 0, nullptr);
        vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dim), &dim);
        vkCmdDispatch(cb, (dim + 255u) / 256u, 1, 1);
    });
    if (st == VG_OK) memcpy(out, vk->y_buf->mapped, (size_t)dim * sizeof(float));
    destroy_buffer(vk, gb);
    destroy_buffer(vk, ub);
    return st;
}

VG_Status vg_vk_rope(VG_VK *vk, float *data, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
                     float theta_base, uint32_t pos) {
    if (!vk || !data || !n_head || !head_dim || !n_rot) return VG_E_INVALID;
    VG_VKPipeline *pl = &vk->rope;
    if (!pl->ready) return VG_E_UNSUPPORTED;
    const uint32_t n = n_head * head_dim;
    VG_Status st = ensure_staging(vk, n, 0);
    if (st != VG_OK) return st;
    memcpy(vk->x_buf->mapped, data, (size_t)n * sizeof(float));
    VG_VKBuffer *bufs[1] = {vk->x_buf};
    bind_buffers(vk, pl->set, bufs, 1);
    uint32_t pc[6] = {head_dim, n_rot, n_head, pos, 0, 0};
    memcpy(&pc[4], &theta_base, sizeof(float));
    pc[5] = (n_rot / 2u) < (head_dim / 2u) ? n_rot / 2u : head_dim / 2u;
    st = submit_recorded(vk, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &pl->set, 0, nullptr);
        vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cb, (n_head * pc[5] + 63u) / 64u, 1, 1);
    });
    if (st == VG_OK) memcpy(data, vk->x_buf->mapped, (size_t)n * sizeof(float));
    return st;
}
