// Numerical prototype runner only; host-coherent buffers are not a performance claim.
#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <cstring>
#include <stdexcept>
inline void vk_ok(VkResult r){if(r!=VK_SUCCESS)throw std::runtime_error("Vulkan prototype failed");}
struct RawCompute {
    VkInstance instance{};VkPhysicalDevice physical{};VkDevice device{};VkQueue queue{};uint32_t family{};
    RawCompute(){
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ci.pApplicationInfo=&app;vk_ok(vkCreateInstance(&ci,nullptr,&instance));
        uint32_t n=0;vk_ok(vkEnumeratePhysicalDevices(instance,&n,nullptr));std::vector<VkPhysicalDevice> all(n);vk_ok(vkEnumeratePhysicalDevices(instance,&n,all.data()));
        for(auto p:all){VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(p,&props);if(props.vendorID==0x1002){physical=p;break;}}
        if(!physical)throw std::runtime_error("AMD prototype device missing");
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&n,nullptr);std::vector<VkQueueFamilyProperties> families(n);vkGetPhysicalDeviceQueueFamilyProperties(physical,&n,families.data());
        bool found=false;for(uint32_t i=0;i<n;++i)if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT){family=i;found=true;break;}if(!found)throw std::runtime_error("Compute queue missing");
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;vk_ok(vkCreateDevice(physical,&di,nullptr,&device));vkGetDeviceQueue(device,family,0,&queue);
    }
    ~RawCompute(){if(device){vkDeviceWaitIdle(device);vkDestroyDevice(device,nullptr);}if(instance)vkDestroyInstance(instance,nullptr);}
    struct Buffer {VkBuffer buffer{};VkDeviceMemory memory{};VkDevice device{};size_t bytes{};void * mapped{};
        Buffer(RawCompute & c,size_t size):device(c.device),bytes(size){
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=size;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;vk_ok(vkCreateBuffer(device,&bi,nullptr,&buffer));
            VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,buffer,&req);VkPhysicalDeviceMemoryProperties mp;vkGetPhysicalDeviceMemoryProperties(c.physical,&mp);uint32_t type=UINT32_MAX;
            for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((req.memoryTypeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
            if(type==UINT32_MAX)throw std::runtime_error("Coherent memory missing");VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=type;vk_ok(vkAllocateMemory(device,&ai,nullptr,&memory));vk_ok(vkBindBufferMemory(device,buffer,memory,0));vk_ok(vkMapMemory(device,memory,0,size,0,&mapped));
        }
        ~Buffer(){if(mapped)vkUnmapMemory(device,memory);if(buffer)vkDestroyBuffer(device,buffer,nullptr);if(memory)vkFreeMemory(device,memory,nullptr);}
    };
    std::vector<float> run(const char * path,const std::vector<std::vector<float>> & values,size_t output,const void * pc,uint32_t pc_size,uint32_t gx,uint32_t gy=1){
        std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("SPIR-V missing");auto size=file.tellg();std::vector<uint32_t> code(size_t(size)/4);file.seekg(0);file.read((char*)code.data(),size);
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};si.codeSize=size_t(size);si.pCode=code.data();VkShaderModule shader;vk_ok(vkCreateShaderModule(device,&si,nullptr,&shader));
        const uint32_t count=uint32_t(values.size()+1);std::vector<Buffer*> buffers;std::vector<VkDescriptorSetLayoutBinding> bindings;std::vector<VkDescriptorBufferInfo> infos;
        for(uint32_t i=0;i<count;++i){auto * p=new Buffer(*this,(i<values.size()?values[i].size():output)*4);buffers.push_back(p);if(i<values.size())std::memcpy(p->mapped,values[i].data(),p->bytes);bindings.push_back({i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr});infos.push_back({p->buffer,0,p->bytes});}
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=count;li.pBindings=bindings.data();VkDescriptorSetLayout layout;vk_ok(vkCreateDescriptorSetLayout(device,&li,nullptr,&layout));
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,pc_size};VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pi.setLayoutCount=1;pi.pSetLayouts=&layout;pi.pushConstantRangeCount=1;pi.pPushConstantRanges=&range;VkPipelineLayout pipeline_layout;vk_ok(vkCreatePipelineLayout(device,&pi,nullptr,&pipeline_layout));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=pipeline_layout;ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=shader;ci.stage.pName="main";VkPipeline pipeline;vk_ok(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline));
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,count};VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&pool_size;VkDescriptorPool pool;vk_ok(vkCreateDescriptorPool(device,&dpi,nullptr,&pool));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=pool;dai.descriptorSetCount=1;dai.pSetLayouts=&layout;VkDescriptorSet set;vk_ok(vkAllocateDescriptorSets(device,&dai,&set));std::vector<VkWriteDescriptorSet> writes(count);
        for(uint32_t i=0;i<count;++i){writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}vkUpdateDescriptorSets(device,count,writes.data(),0,nullptr);
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=family;VkCommandPool commands;vk_ok(vkCreateCommandPool(device,&cpi,nullptr,&commands));VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=commands;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;VkCommandBuffer command;vk_ok(vkAllocateCommandBuffers(device,&cai,&command));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};vk_ok(vkBeginCommandBuffer(command,&begin));vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&set,0,nullptr);vkCmdPushConstants(command,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,pc_size,pc);vkCmdDispatch(command,gx,gy,1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);vk_ok(vkEndCommandBuffer(command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;vk_ok(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE));vk_ok(vkQueueWaitIdle(queue));std::vector<float> result(output);std::memcpy(result.data(),buffers.back()->mapped,output*4);
        vkDestroyCommandPool(device,commands,nullptr);vkDestroyDescriptorPool(device,pool,nullptr);vkDestroyPipeline(device,pipeline,nullptr);vkDestroyPipelineLayout(device,pipeline_layout,nullptr);vkDestroyDescriptorSetLayout(device,layout,nullptr);vkDestroyShaderModule(device,shader,nullptr);for(auto p:buffers)delete p;return result;
    }
};
