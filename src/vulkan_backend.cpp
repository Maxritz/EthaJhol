#include "vg/vulkan_backend.h"
#include "vg/trace.h"

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

struct VG_VKBuffer { VkBuffer buffer{}; VkDeviceMemory memory{}; void *mapped{}; size_t bytes{}; };

struct VG_VKPipeline {
    VkPipeline pipeline{};
    VkPipelineLayout layout{};
    VkDescriptorSetLayout set_layout{};
    uint32_t push_size{};
};

struct VG_VK {
    VkInstance instance{}; VkPhysicalDevice physical{}; VkDevice device{}; VkQueue queue{}; uint32_t family{};
    VkCommandPool command_pool{};
    VkPhysicalDeviceMemoryProperties mem{}; VG_VKInfo info{}; std::string shader_dir;
    uint64_t memory_budget_bytes{}; uint64_t memory_allocated{};

    VG_VKPipeline matvec_i8;
    VG_VKPipeline matvec_f16;
    VG_VKPipeline matvec_f32;
    VG_VKPipeline matvec_q4_0;
    VG_VKPipeline matvec_q8_0;
    VG_VKPipeline rmsnorm;
    VG_VKPipeline rope;
    VG_VKPipeline softmax;
    VG_VKPipeline swiglu;
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
static VG_Status make_buffer(VG_VK *v, size_t bytes, VkBufferUsageFlags usage, VG_VKBuffer **out) {
    if (!v || !out || bytes == 0) return VG_E_INVALID; *out = nullptr;
    if (v->memory_allocated + bytes > v->memory_budget_bytes) return VG_E_NOMEM;
    VG_VKBuffer *b = new VG_VKBuffer(); b->bytes = bytes;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = bytes; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(v->device, &ci, nullptr, &b->buffer) != VK_SUCCESS) { delete b; return VG_E_NOMEM; }
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(v->device, b->buffer, &req);
    uint32_t ti = mem_type(v->mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (ti == UINT32_MAX) ti = mem_type(v->mem, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (ti == UINT32_MAX) { destroy_buffer(v, b); return VG_E_UNSUPPORTED; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = ti;
    if (vkAllocateMemory(v->device, &ai, nullptr, &b->memory) != VK_SUCCESS || vkBindBufferMemory(v->device, b->buffer, b->memory, 0) != VK_SUCCESS || vkMapMemory(v->device, b->memory, 0, req.size, 0, &b->mapped) != VK_SUCCESS) { destroy_buffer(v, b); return VG_E_NOMEM; }
    *out = b; v->memory_allocated += bytes; return VG_OK;
}
static VG_Status submit_one(VG_VK *v, const std::function<void(VkCommandBuffer)> &record) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool = v->command_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1; VkCommandBuffer cb{};
    if (vkAllocateCommandBuffers(v->device, &ai, &cb) != VK_SUCCESS) return VG_E_NOMEM;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; VG_Status st = VG_OK;
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) st = VG_E_IO; else { record(cb); if (vkEndCommandBuffer(cb) != VK_SUCCESS) st = VG_E_IO; }
    if (st == VG_OK) { VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; if (vkQueueSubmit(v->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(v->queue) != VK_SUCCESS) st = VG_E_IO; }
    vkFreeCommandBuffers(v->device, v->command_pool, 1, &cb); return st;
}

static void destroy_pipeline(VG_VK *v, VG_VKPipeline *p) {
    if (!p) return;
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
    return VG_OK;
}

static VG_Status alloc_desc_set(VG_VK *v, VkDescriptorSetLayout layout, VkDescriptorPool *pool, VkDescriptorSet *set) {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
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
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(v->physical, &qn, nullptr); std::vector<VkQueueFamilyProperties> qs(qn); vkGetPhysicalDeviceQueueFamilyProperties(v->physical, &qn, qs.data()); bool found = false; for (uint32_t i = 0; i < qn; ++i) if (qs[i].queueCount && (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { v->family = i; found = true; break; } if (!found) { vg_vk_close(v); return VG_E_UNSUPPORTED; }
    float priority = 1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = v->family; qci.queueCount = 1; qci.pQueuePriorities = &priority; VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(v->physical, &dci, nullptr, &v->device) != VK_SUCCESS) { vg_vk_close(v); return VG_E_UNSUPPORTED; } vkGetDeviceQueue(v->device, v->family, 0, &v->queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = v->family; pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; if (vkCreateCommandPool(v->device, &pci, nullptr, &v->command_pool) != VK_SUCCESS) { vg_vk_close(v); return VG_E_NOMEM; }

    VG_Status s;
    s = create_pipeline(v, &v->matvec_i8, "matvec_i8.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->matvec_f16, "matvec_f16.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->matvec_f32, "matvec_f32.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->matvec_q4_0, "matvec_q4_0.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->matvec_q8_0, "matvec_q8_0.comp.spv", 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->rmsnorm, "rmsnorm.comp.spv", 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->rope, "rope.comp.spv", 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->softmax, "softmax.comp.spv", 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4);
    if (s != VG_OK) { vg_vk_close(v); return s; }
    s = create_pipeline(v, &v->swiglu, "swiglu.comp.spv", 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4);
    if (s != VG_OK) { vg_vk_close(v); return s; }

    *out = v; return VG_OK;
}

void vg_vk_close(VG_VK *v) {
    if (!v) return;
    if (v->device) vkDeviceWaitIdle(v->device);
    destroy_pipeline(v, &v->matvec_i8);
    destroy_pipeline(v, &v->matvec_f16);
    destroy_pipeline(v, &v->matvec_f32);
    destroy_pipeline(v, &v->matvec_q4_0);
    destroy_pipeline(v, &v->matvec_q8_0);
    destroy_pipeline(v, &v->rmsnorm);
    destroy_pipeline(v, &v->rope);
    destroy_pipeline(v, &v->softmax);
    destroy_pipeline(v, &v->swiglu);
    if (v->command_pool) vkDestroyCommandPool(v->device, v->command_pool, nullptr);
    if (v->device) vkDestroyDevice(v->device, nullptr);
    if (v->instance) vkDestroyInstance(v->instance, nullptr);
    delete v;
}

VG_Status vg_vk_info(const VG_VK *v, VG_VKInfo *out) { if (!v || !out) return VG_E_INVALID; *out = v->info; return VG_OK; }
VG_Status vg_vk_buffer_upload(VG_VK *v, const void *data, size_t bytes, VG_VKBuffer **out) { VG_Status s = make_buffer(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, out); if (s != VG_OK) return s; std::memcpy((*out)->mapped, data, bytes); VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = (*out)->memory; r.size = VK_WHOLE_SIZE; vkFlushMappedMemoryRanges(v->device, 1, &r); return VG_OK; }
VG_Status vg_vk_buffer_read(VG_VK *v, const VG_VKBuffer *b, void *data, size_t bytes) { if (!v || !b || !data || bytes > b->bytes) return VG_E_INVALID; VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; r.memory = b->memory; r.size = VK_WHOLE_SIZE; vkInvalidateMappedMemoryRanges(v->device, 1, &r); std::memcpy(data, b->mapped, bytes); return VG_OK; }
void vg_vk_buffer_release(VG_VK *v, VG_VKBuffer *b) { destroy_buffer(v, b); }

static VG_Status dispatch_matvec(VG_VK *v, VG_VKPipeline *pl, const VG_VKBuffer *weights, const VG_VKBuffer *scales, const float *x, float *y, uint32_t rows, uint32_t input, uint32_t output) {
    if (!v || !weights || !scales || !x || !y || !rows || !input || !output) return VG_E_INVALID;
    uint64_t t0 = vg_trace_now_ns();
    VG_VKBuffer *xb = nullptr, *yb = nullptr;
    VG_Status s = vg_vk_buffer_upload(v, x, static_cast<size_t>(rows) * input * sizeof(float), &xb);
    if (s != VG_OK) return s;
    s = make_buffer(v, static_cast<size_t>(rows) * output * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &yb);
    if (s != VG_OK) { destroy_buffer(v, xb); return s; }

    VkDescriptorPool pool{}; VkDescriptorSet set{};
    s = alloc_desc_set(v, pl->set_layout, &pool, &set);
    if (s != VG_OK) { destroy_buffer(v, xb); destroy_buffer(v, yb); return s; }

    VG_VKBuffer *bufs[4] = {(VG_VKBuffer *)weights, (VG_VKBuffer *)scales, xb, yb};
    bind_buffers(v, set, bufs, 4);

    s = submit_one(v, [&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout, 0, 1, &set, 0, nullptr);
        uint32_t p[3] = {rows, input, output};
        vkCmdPushConstants(cb, pl->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), p);
        uint32_t total = rows * output;
        vkCmdDispatch(cb, (total + 63u) / 64u, 1, 1);
    });
    if (s == VG_OK) s = vg_vk_buffer_read(v, yb, y, static_cast<size_t>(rows) * output * sizeof(float));
    vg_trace_emit("vk_matvec", t0, vg_trace_now_ns(), static_cast<size_t>(rows) * output * sizeof(float), 2);
    vkDestroyDescriptorPool(v->device, pool, nullptr);
    destroy_buffer(v, xb); destroy_buffer(v, yb);
    return s;
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
    return dispatch_matvec(v, &v->matvec_q8_0, weights, scales, x, y, rows, input, output);
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
