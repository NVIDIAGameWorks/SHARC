/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define SHARC_VERSION_MAJOR                 1
#define SHARC_VERSION_MINOR                 3
#define SHARC_VERSION_BUILD                 1
#define SHARC_VERSION_REVISION              0

#define SHARC_SAMPLE_NUM_MULTIPLIER             16      // increase sample count internally to make resolve step with low sample count more robust, power of 2 usage may help compiler with optimizations
#define SHARC_SAMPLE_NUM_THRESHOLD              0       // elements with sample count above this threshold will be used for early-out/resampling
#define SHARC_SEPARATE_EMISSIVE                 0       // if set, emissive values should be passed separately on updates and added to the cache query
#define SHARC_PROPOGATION_DEPTH                 4       // controls the amount of vertices stored in memory for signal backpropagation
#define SHARC_ENABLE_CACHE_RESAMPLING           (SHARC_PROPOGATION_DEPTH > 1) // resamples the cache during update step
#define SHARC_RESAMPLING_DEPTH_MIN              1       // controls minimum path depth which can be used with cache resampling
#define SHARC_RADIANCE_SCALE                    1e3f    // scale used for radiance values accumulation. Each component uses 32-bit integer for data storage
#define SHARC_ACCUMULATED_FRAME_NUM_MIN         1       // minimum number of frames to use for data accumulation
#define SHARC_ACCUMULATED_FRAME_NUM_MAX         64      // maximum number of frames to use for data accumulation
#define SHARC_STALE_FRAME_NUM_MIN               32      // minimum number of frames to keep the element in the cache
#define SHARC_SAMPLE_NUM_BIT_NUM                18
#define SHARC_SAMPLE_NUM_BIT_OFFSET             0
#define SHARC_SAMPLE_NUM_BIT_MASK               ((1u << SHARC_SAMPLE_NUM_BIT_NUM) - 1)
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM     6
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET  (SHARC_SAMPLE_NUM_BIT_NUM)
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK    ((1u << SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM) - 1)
#define SHARC_STALE_FRAME_NUM_BIT_NUM           8
#define SHARC_STALE_FRAME_NUM_BIT_OFFSET        (SHARC_SAMPLE_NUM_BIT_NUM + SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM)
#define SHARC_STALE_FRAME_NUM_BIT_MASK          ((1u << SHARC_STALE_FRAME_NUM_BIT_NUM) - 1)
#define SHARC_GRID_LOGARITHM_BASE               2.0f
#define SHARC_ENABLE_COMPACTION                 HASH_GRID_ALLOW_COMPACTION
#define SHARC_BLEND_ADJACENT_LEVELS             1       // combine the data from adjacent levels on camera movement
#define SHARC_DEFERRED_HASH_COMPACTION          (SHARC_ENABLE_COMPACTION && SHARC_BLEND_ADJACENT_LEVELS)
#define SHARC_NORMALIZED_SAMPLE_NUM             (1u << (SHARC_SAMPLE_NUM_BIT_NUM - 1))

// Debug
#define SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_LOW        0.125
#define SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_MEDIUM     0.5

#if SHARC_ENABLE_GLSL

// Required extensions
// #extension GL_EXT_buffer_reference : require
// #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// #extension GL_EXT_shader_atomic_int64 : require
// #extension GL_KHR_shader_subgroup_ballot : require

// Buffer reference types can be constructed from a 'uint64_t' or a 'uvec2' value.
// The low - order 32 bits of the reference map to and from the 'x' component
// of the 'uvec2'.

#define float2 vec2
#define float3 vec3
#define float4 vec4

#define uint2 uvec2
#define uint3 uvec3
#define uint4 uvec4

#define int2 ivec2
#define int3 ivec3
#define int4 ivec4

#define lerp mix
#define InterlockedAdd atomicAdd
#define InterlockedCompareExchange atomicCompSwap
#define WaveActiveCountBits(value) subgroupBallotBitCount(uint4(value, 0, 0, 0))
#define WaveActiveBallot subgroupBallot
#define WavePrefixCountBits(value) subgroupBallotExclusiveBitCount(uint4(value, 0, 0, 0))

#define RW_STRUCTURED_BUFFER(name, type) RWStructuredBuffer_##type name
#define BUFFER_AT_OFFSET(name, offset) name.data[offset]

layout(buffer_reference, std430, buffer_reference_align = 8) buffer RWStructuredBuffer_uint64_t {
    uint64_t data[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer RWStructuredBuffer_uint {
    uint data[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer RWStructuredBuffer_uint4 {
    uvec4 data[];
};

#else // !SHARC_ENABLE_GLSL

#define RW_STRUCTURED_BUFFER(name, type) RWStructuredBuffer<type> name
#define BUFFER_AT_OFFSET(name, offset) name[offset]

#endif // !SHARC_ENABLE_GLSL

/*
 * RTXGI2 DIVERGENCE:
 *    Use SHARC_ENABLE_64_BIT_ATOMICS instead of SHARC_DISABLE_64_BIT_ATOMICS
 *    (Prefer 'enable' bools over 'disable' to avoid unnecessary mental gymnastics)
 *    Automatically set SHARC_ENABLE_64_BIT_ATOMICS if we're using DXC and it's not defined.
 */
#if !defined(SHARC_ENABLE_64_BIT_ATOMICS) && defined(__DXC_VERSION_MAJOR)
// Use DXC macros to figure out if 64-bit atomics are possible from the current shader model
#if __SHADER_TARGET_MAJOR < 6
#define SHARC_ENABLE_64_BIT_ATOMICS 0
#elif __SHADER_TARGET_MAJOR > 6
#define SHARC_ENABLE_64_BIT_ATOMICS 1
#else
// 6.x
#if __SHADER_TARGET_MINOR < 6
#define SHARC_ENABLE_64_BIT_ATOMICS 0
#else
#define SHARC_ENABLE_64_BIT_ATOMICS 1
#endif
#endif
#elif !defined(SHARC_ENABLE_64_BIT_ATOMICS)
// Not DXC, and SHARC_ENABLE_64_BIT_ATOMICS not defined
#error "Please define SHARC_ENABLE_64_BIT_ATOMICS as 0 or 1"
#endif

#if SHARC_ENABLE_64_BIT_ATOMICS
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#else
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 0
#endif
#include "HashGridCommon.h"

struct SharcVoxelData
{
    uint3 accumulatedRadiance;
    uint accumulatedSampleNum;
    uint accumulatedFrameNum;
    uint staleFrameNum;
};

uint SharcGetSampleNum(uint packedData)
{
    return (packedData >> SHARC_SAMPLE_NUM_BIT_OFFSET) & SHARC_SAMPLE_NUM_BIT_MASK;
}

uint SharcGetStaleFrameNum(uint packedData)
{
    return (packedData >> SHARC_STALE_FRAME_NUM_BIT_OFFSET) & SHARC_STALE_FRAME_NUM_BIT_MASK;
}

uint SharcGetAccumulatedFrameNum(uint packedData)
{
    return (packedData >> SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET) & SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK;
}

float3 SharcResolveAccumulatedRadiance(uint3 accumulatedRadiance, uint accumulatedSampleNum)
{
    return accumulatedRadiance / (accumulatedSampleNum * float(SHARC_RADIANCE_SCALE));
}

SharcVoxelData SharcUnpackVoxelData(uint4 voxelDataPacked)
{
    SharcVoxelData voxelData;
    voxelData.accumulatedRadiance = voxelDataPacked.xyz;
    voxelData.accumulatedSampleNum = SharcGetSampleNum(voxelDataPacked.w);
    voxelData.staleFrameNum = SharcGetStaleFrameNum(voxelDataPacked.w);
    voxelData.accumulatedFrameNum = SharcGetAccumulatedFrameNum(voxelDataPacked.w);
    return voxelData;
}

SharcVoxelData SharcGetVoxelData(RW_STRUCTURED_BUFFER(voxelDataBuffer, uint4), CacheEntry cacheEntry)
{
    SharcVoxelData voxelData;
    voxelData.accumulatedRadiance = uint3(0, 0, 0);
    voxelData.accumulatedSampleNum = 0;
    voxelData.accumulatedFrameNum = 0;
    voxelData.staleFrameNum = 0;

    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY)
        return voxelData;

    uint4 voxelDataPacked = BUFFER_AT_OFFSET(voxelDataBuffer, cacheEntry);

    return SharcUnpackVoxelData(voxelDataPacked);
}

void SharcAddVoxelData(RW_STRUCTURED_BUFFER(voxelDataBuffer, uint4), CacheEntry cacheEntry, float3 value, uint sampleData)
{
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY)
        return;

    uint3 scaledRadiance = uint3(value * SHARC_RADIANCE_SCALE);

    if (scaledRadiance.x != 0) InterlockedAdd(BUFFER_AT_OFFSET(voxelDataBuffer, cacheEntry).x, scaledRadiance.x);
    if (scaledRadiance.y != 0) InterlockedAdd(BUFFER_AT_OFFSET(voxelDataBuffer, cacheEntry).y, scaledRadiance.y);
    if (scaledRadiance.z != 0) InterlockedAdd(BUFFER_AT_OFFSET(voxelDataBuffer, cacheEntry).z, scaledRadiance.z);
    if (sampleData != 0) InterlockedAdd(BUFFER_AT_OFFSET(voxelDataBuffer, cacheEntry).w, sampleData);
}

struct SharcPathPayload
{
    CacheEntry cacheEntry[SHARC_PROPOGATION_DEPTH];
    float3 sampleWeight[SHARC_PROPOGATION_DEPTH];
    uint pathLength;
};

struct SharcState
{
    GridParameters gridParameters;
    HashMapData hashMapData;

    RW_STRUCTURED_BUFFER(voxelDataBuffer, uint4);

#if SHARC_ENABLE_CACHE_RESAMPLING
    RW_STRUCTURED_BUFFER(voxelDataBufferPrev, uint4);
#endif // SHARC_ENABLE_CACHE_RESAMPLING
};

struct SharcHitData
{
    float3 positionWorld;
    float3 normalWorld;
#if SHARC_SEPARATE_EMISSIVE
    float3 emissive;
#endif // SHARC_SEPARATE_EMISSIVE
};

void SharcPayloadInit(inout SharcPathPayload sharcPayload)
{
    sharcPayload.pathLength = 0;
}

void SharcPayloadSetThroughput(inout SharcPathPayload sharcPayload, float3 throughput)
{
    sharcPayload.sampleWeight[0] = throughput;
}

void SharcUpdateMiss(inout SharcPathPayload sharcPayload, SharcState sharcState, float3 radiance)
{
    for (int i = 0; i < sharcPayload.pathLength; ++i)
    {
        radiance *= sharcPayload.sampleWeight[i];
        SharcAddVoxelData(sharcState.voxelDataBuffer, sharcPayload.cacheEntry[i], radiance, 0);
    }
}

bool SharcUpdateHit(inout SharcPathPayload sharcPayload, SharcState sharcState, SharcHitData sharcHitData, float3 lighting, float random)
{
    bool continueTracing = true;

    CacheEntry cacheEntry = HashMapInsertEntry(sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);

    float3 sharcRadiance = lighting;

#if SHARC_ENABLE_CACHE_RESAMPLING
    uint resamplingDepth = uint(round(lerp(SHARC_RESAMPLING_DEPTH_MIN, SHARC_PROPOGATION_DEPTH - 1, random)));
    if (resamplingDepth <= sharcPayload.pathLength)
    {
        SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBufferPrev, cacheEntry);
        if (voxelData.accumulatedSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
        {
            sharcRadiance = SharcResolveAccumulatedRadiance(voxelData.accumulatedRadiance, voxelData.accumulatedSampleNum);
            continueTracing = false;
        }
    }
#endif // SHARC_ENABLE_CACHE_RESAMPLING

    if (continueTracing)
        SharcAddVoxelData(sharcState.voxelDataBuffer, cacheEntry, lighting, 1);

#if SHARC_SEPARATE_EMISSIVE
    sharcRadiance += sharcHitData.emissive;
#endif // SHARC_SEPARATE_EMISSIVE

    uint i;
    for (i = 0; i < sharcPayload.pathLength; ++i)
    {
        sharcRadiance *= sharcPayload.sampleWeight[i];
        SharcAddVoxelData(sharcState.voxelDataBuffer, sharcPayload.cacheEntry[i], sharcRadiance, 0);
    }

    for (i = sharcPayload.pathLength; i > 0; --i)
    {
        sharcPayload.cacheEntry[i] = sharcPayload.cacheEntry[i - 1];
        sharcPayload.sampleWeight[i] = sharcPayload.sampleWeight[i - 1];
    }

    sharcPayload.cacheEntry[0] = cacheEntry;
    sharcPayload.pathLength = min(++sharcPayload.pathLength, SHARC_PROPOGATION_DEPTH - 1);

    return continueTracing;
}

bool SharcGetCachedRadiance(in SharcState sharcState, in SharcHitData sharcHitData, out float3 radiance, bool debug)
{
    if (debug) radiance = float3(0, 0, 0);
    const uint sampleThreshold = debug ? 0 : SHARC_SAMPLE_NUM_THRESHOLD;

    CacheEntry cacheEntry = HashMapFindEntry(sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY)
        return false;

    SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBuffer, cacheEntry);
    if (voxelData.accumulatedSampleNum > sampleThreshold)
    {
        radiance = SharcResolveAccumulatedRadiance(voxelData.accumulatedRadiance, voxelData.accumulatedSampleNum);

#if SHARC_SEPARATE_EMISSIVE
        radiance += sharcHitData.emissive;
#endif // SHARC_SEPARATE_EMISSIVE

        return true;
    }

    return false;
}

void SharcCopyHashEntry(uint entryIndex, HashMapData hashMapData, RW_STRUCTURED_BUFFER(copyOffsetBuffer, uint))
{
#if SHARC_DEFERRED_HASH_COMPACTION
    if (entryIndex >= hashMapData.capacity)
        return;

    uint copyOffset = BUFFER_AT_OFFSET(copyOffsetBuffer, entryIndex);
    if (copyOffset == 0)
        return;

    if (copyOffset == HASH_GRID_INVALID_CACHE_ENTRY)
    {
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex) = HASH_GRID_INVALID_HASH_KEY;
    }
    else if (copyOffset != 0)
    {
        HashKey hashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex);
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex) = HASH_GRID_INVALID_HASH_KEY;
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, copyOffset) = hashKey;
    }

    BUFFER_AT_OFFSET(copyOffsetBuffer, entryIndex) = 0;
#endif // SHARC_DEFERRED_HASH_COMPACTION
}

int SharcGetGridDistance2(int3 position)
{
    return position.x * position.x + position.y * position.y + position.z * position.z;
}

HashKey SharcGetAdjacentLevelHashKey(HashKey hashKey, GridParameters gridParameters)
{
	const int signBit      = 1 << (HASH_GRID_POSITION_BIT_NUM - 1);
    const int signMask     = ~((1 << HASH_GRID_POSITION_BIT_NUM) - 1);

    int3 gridPosition;
    gridPosition.x = int((hashKey >> HASH_GRID_POSITION_BIT_NUM * 0) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.y = int((hashKey >> HASH_GRID_POSITION_BIT_NUM * 1) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.z = int((hashKey >> HASH_GRID_POSITION_BIT_NUM * 2) & HASH_GRID_POSITION_BIT_MASK);

    // Fix negative coordinates
    gridPosition.x = ((gridPosition.x & signBit) != 0) ? gridPosition.x | signMask : gridPosition.x;
    gridPosition.y = ((gridPosition.y & signBit) != 0) ? gridPosition.y | signMask : gridPosition.y;
    gridPosition.z = ((gridPosition.z & signBit) != 0) ? gridPosition.z | signMask : gridPosition.z;

    int level = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 3)) & HASH_GRID_LEVEL_BIT_MASK);

    float voxelSize = GetVoxelSize(level, gridParameters);
    int3 cameraGridPosition = int3(floor((gridParameters.cameraPosition + HASH_GRID_POSITION_OFFSET) / voxelSize));
    int3 cameraVector = cameraGridPosition - gridPosition;
    int cameraDistance = SharcGetGridDistance2(cameraVector);

    int3 cameraGridPositionPrev = int3(floor((gridParameters.cameraPositionPrev + HASH_GRID_POSITION_OFFSET) / voxelSize));
    int3 cameraVectorPrev = cameraGridPositionPrev - gridPosition;
    int cameraDistancePrev = SharcGetGridDistance2(cameraVectorPrev);

    if (cameraDistance < cameraDistancePrev)
    {
        gridPosition = int3(floor(gridPosition / gridParameters.logarithmBase));
        level = min(level + 1, int(HASH_GRID_LEVEL_BIT_MASK));
    }
    else // this may be inaccurate
    {
        gridPosition = int3(floor(gridPosition * gridParameters.logarithmBase));
        level = max(level - 1, 1);
    }

    HashKey modifiedHashKey = ((uint64_t(gridPosition.x) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0))
        | ((uint64_t(gridPosition.y) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1))
        | ((uint64_t(gridPosition.z) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2))
        | ((uint64_t(level) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

#if HASH_GRID_USE_NORMALS
    modifiedHashKey |= hashKey & (uint64_t(HASH_GRID_NORMAL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
#endif // HASH_GRID_USE_NORMALS

    return modifiedHashKey;
}

void SharcResolveEntry(uint entryIndex, GridParameters gridParameters, HashMapData hashMapData, RW_STRUCTURED_BUFFER(copyOffsetBuffer, uint),
    RW_STRUCTURED_BUFFER(voxelDataBuffer, uint4), RW_STRUCTURED_BUFFER(voxelDataBufferPrev, uint4), uint accumulationFrameNum, uint staleFrameNumMax)
{
    if (entryIndex >= hashMapData.capacity)
        return;

    HashKey hashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex);
    if (hashKey == HASH_GRID_INVALID_HASH_KEY)
        return;

    uint4 voxelDataPackedPrev = BUFFER_AT_OFFSET(voxelDataBufferPrev, entryIndex);
    uint4 voxelDataPacked = BUFFER_AT_OFFSET(voxelDataBuffer, entryIndex);

    uint sampleNum = SharcGetSampleNum(voxelDataPacked.w);
    uint sampleNumPrev = SharcGetSampleNum(voxelDataPackedPrev.w);
    uint accumulatedFrameNum = SharcGetAccumulatedFrameNum(voxelDataPackedPrev.w);
    uint staleFrameNum = SharcGetStaleFrameNum(voxelDataPackedPrev.w);

    uint3 accumulatedRadiance = voxelDataPacked.xyz * SHARC_SAMPLE_NUM_MULTIPLIER + voxelDataPackedPrev.xyz;
    uint accumulatedSampleNum = SharcGetSampleNum(voxelDataPacked.w) * SHARC_SAMPLE_NUM_MULTIPLIER + SharcGetSampleNum(voxelDataPackedPrev.w);

#if SHARC_BLEND_ADJACENT_LEVELS
    // Reproject sample from adjacent level
    float3 cameraOffset = gridParameters.cameraPosition.xyz - gridParameters.cameraPositionPrev.xyz;
    if ((dot(cameraOffset, cameraOffset) != 0) && (accumulatedFrameNum < accumulationFrameNum))
    {
        HashKey adjacentLevelHashKey = SharcGetAdjacentLevelHashKey(hashKey, gridParameters);

        CacheEntry cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
        if (HashMapFind(hashMapData, adjacentLevelHashKey, cacheEntry))
        {
            uint4 adjacentPackedDataPrev = BUFFER_AT_OFFSET(voxelDataBufferPrev, cacheEntry);
            uint adjacentSampleNum = SharcGetSampleNum(adjacentPackedDataPrev.w);
            if (adjacentSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
            {
                float blendWeight = adjacentSampleNum / float(adjacentSampleNum + accumulatedSampleNum);
                accumulatedRadiance = uint3(lerp(float3(accumulatedRadiance.xyz), float3(adjacentPackedDataPrev.xyz), blendWeight));
                accumulatedSampleNum = uint(lerp(float(accumulatedSampleNum), float(adjacentSampleNum), blendWeight));
            }
        }
    }
#endif // SHARC_BLEND_ADJACENT_LEVELS

    // Clamp internal sample count to help with potential overflow
    if (accumulatedSampleNum > SHARC_NORMALIZED_SAMPLE_NUM)
    {
        accumulatedSampleNum >>= 1;
        accumulatedRadiance >>= 1;
    }

    accumulationFrameNum = clamp(accumulationFrameNum, SHARC_ACCUMULATED_FRAME_NUM_MIN, SHARC_ACCUMULATED_FRAME_NUM_MAX);
    if (accumulatedFrameNum > accumulationFrameNum)
    {
        float normalizedAccumulatedSampleNum = round(accumulatedSampleNum * float(accumulationFrameNum) / accumulatedFrameNum);
        float normalizationScale = normalizedAccumulatedSampleNum / accumulatedSampleNum;

        accumulatedSampleNum = uint(normalizedAccumulatedSampleNum);
        accumulatedRadiance = uint3(accumulatedRadiance * normalizationScale);
        accumulatedFrameNum = uint(accumulatedFrameNum * normalizationScale);
    }

    ++accumulatedFrameNum;
    staleFrameNum = (sampleNum != 0) ? 0 : staleFrameNum + 1;

    uint4 packedData;
    packedData.xyz = accumulatedRadiance;

    packedData.w = min(accumulatedSampleNum, SHARC_SAMPLE_NUM_BIT_MASK);
    packedData.w |= (min(accumulatedFrameNum, SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK) << SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET);
    packedData.w |= (min(staleFrameNum, SHARC_STALE_FRAME_NUM_BIT_MASK) << SHARC_STALE_FRAME_NUM_BIT_OFFSET);

    bool isValidElement = (staleFrameNum < max(staleFrameNumMax, SHARC_STALE_FRAME_NUM_MIN)) ? true : false;

    if (!isValidElement)
    {
        packedData = uint4(0, 0, 0, 0);
#if !SHARC_ENABLE_COMPACTION
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex) = HASH_GRID_INVALID_HASH_KEY;
#endif // !SHARC_ENABLE_COMPACTION
    }

#if SHARC_ENABLE_COMPACTION
    uint validElementNum = WaveActiveCountBits(isValidElement);
    uint validElementMask = WaveActiveBallot(isValidElement).x;
    bool isMovableElement = isValidElement && ((entryIndex % HASH_GRID_HASH_MAP_BUCKET_SIZE) >= validElementNum);
    uint movableElementIndex = WavePrefixCountBits(isMovableElement);

    if ((entryIndex % HASH_GRID_HASH_MAP_BUCKET_SIZE) >= validElementNum)
    {
        uint writeOffset = 0;
#if !SHARC_DEFERRED_HASH_COMPACTION
        hashMapData.hashEntriesBuffer[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
#endif // !SHARC_DEFERRED_HASH_COMPACTION

        BUFFER_AT_OFFSET(voxelDataBuffer, entryIndex) = uint4(0, 0, 0, 0);

        if (isValidElement)
        {
            uint emptySlotIndex = 0;
            while (emptySlotIndex < validElementNum)
            {
                if (((validElementMask >> writeOffset) & 0x1) == 0)
                {
                    if (emptySlotIndex == movableElementIndex)
                    {
                        writeOffset += GetBaseSlot(entryIndex, hashMapData.capacity);
#if !SHARC_DEFERRED_HASH_COMPACTION
                        hashMapData.hashEntriesBuffer[writeOffset] = hashKey;
#endif // !SHARC_DEFERRED_HASH_COMPACTION

                        BUFFER_AT_OFFSET(voxelDataBuffer, writeOffset) = packedData;
                        break;
                    }

                    ++emptySlotIndex;
                }

                ++writeOffset;
            }
        }

#if SHARC_DEFERRED_HASH_COMPACTION
        BUFFER_AT_OFFSET(copyOffsetBuffer, entryIndex) = (writeOffset != 0) ? writeOffset : HASH_GRID_INVALID_CACHE_ENTRY;
#endif // SHARC_DEFERRED_HASH_COMPACTION
    }
    else if (isValidElement)
#endif // SHARC_ENABLE_COMPACTION
    {
        BUFFER_AT_OFFSET(voxelDataBuffer, entryIndex) = packedData;
    }

#if !SHARC_BLEND_ADJACENT_LEVELS
    // Clear buffer entry for the next frame
    //BUFFER_AT_OFFSET(voxelDataBufferPrev, entryIndex) = uint4(0, 0, 0, 0);
#endif // !SHARC_BLEND_ADJACENT_LEVELS
}

// Debug functions
float3 SharcDebugGetBitsOccupancyColor(float occupancy)
{
    if (occupancy < SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_LOW)
        return float3(0.0f, 1.0f, 0.0f) * (occupancy + SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_LOW);
    else if (occupancy < SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_MEDIUM)
        return float3(1.0f, 1.0f, 0.0f) * (occupancy + SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_MEDIUM);
    else
        return float3(1.0f, 0.0f, 0.0f) * occupancy;
}

// Debug visualization
float3 SharcDebugBitsOccupancySampleNum(in SharcState sharcState, in SharcHitData sharcHitData)
{
    CacheEntry cacheEntry = HashMapFindEntry(sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);
    SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBuffer, cacheEntry);

    float occupancy = float(voxelData.accumulatedSampleNum) / SHARC_SAMPLE_NUM_BIT_MASK;

    return SharcDebugGetBitsOccupancyColor(occupancy);
}

float3 SharcDebugBitsOccupancyRadiance(in SharcState sharcState, in SharcHitData sharcHitData)
{
    CacheEntry cacheEntry = HashMapFindEntry(sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);
    SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBuffer, cacheEntry);

    float occupancy = float(max(voxelData.accumulatedRadiance.x, max(voxelData.accumulatedRadiance.y, voxelData.accumulatedRadiance.z))) / 0xffffffff;

    return SharcDebugGetBitsOccupancyColor(occupancy);
}
