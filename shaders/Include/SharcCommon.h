/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define SHARC_SAMPLE_NUM_MAX                128
#define SHARC_SAMPLE_NUM_THRESHOLD          8
#define SHARC_SEPARATE_EMISSIVE             0
#define SHARC_PROPOGATION_DEPTH             4
#define SHARC_ENABLE_CACHE_RESAMPLING       SHARC_UPDATE
#define SHARC_RESAMPLING_DEPTH_MIN          1
#define SHARC_RADIANCE_SCALE                1e4f
#define SHARC_SAMPLE_COUNTER_BIT_NUM        20
#define SHARC_SAMPLE_COUNTER_BIT_MASK       ((1u << SHARC_SAMPLE_COUNTER_BIT_NUM) - 1)
#define SHARC_FRAME_COUNTER_BIT_NUM         (32 - SHARC_SAMPLE_COUNTER_BIT_NUM)
#define SHARC_FRAME_COUNTER_BIT_MASK        ((1u << SHARC_FRAME_COUNTER_BIT_NUM) - 1)
#define SHARC_GRID_LOGARITHM_BASE           2.0f
#define SHARC_STALE_FRAME_NUM_MAX           128
#define SHARC_ENABLE_COMPACTION             HASH_GRID_ALLOW_COMPACTION
#define SHARC_FILTER_ADJACENT_LEVELS        1
#define SHARC_DEFERRED_HASH_COMPACTION      SHARC_ENABLE_COMPACTION

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
#error "Please define SHARC_ENABLE_64_BIT_ATOMICS as 0 or 1
#endif

#if SHARC_ENABLE_64_BIT_ATOMICS
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#else
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 0
#endif
#include "HashGridCommon.h"

struct SharcVoxelData
{
    float3 radiance;
    uint sampleNum;
    uint frameNum;
};

SharcVoxelData SharcUnpackVoxelData(uint4 voxelDataPacked)
{
    SharcVoxelData voxelData = (SharcVoxelData)0;
    voxelData.radiance = voxelDataPacked.xyz / SHARC_RADIANCE_SCALE;
    voxelData.sampleNum = (voxelDataPacked.w >> 0) & SHARC_SAMPLE_COUNTER_BIT_MASK;
    voxelData.frameNum = (voxelDataPacked.w >> SHARC_SAMPLE_COUNTER_BIT_NUM) & SHARC_FRAME_COUNTER_BIT_MASK;

    return voxelData;
}

SharcVoxelData SharcGetVoxelData(RWByteAddressBuffer voxelDataBuffer, CacheEntry cacheEntry)
{
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY)
        return (SharcVoxelData)0;

    uint4 voxelDataPacked = voxelDataBuffer.Load<uint4>(cacheEntry * 16);

    return SharcUnpackVoxelData(voxelDataPacked);
}

void SharcAddVoxelData(RWByteAddressBuffer voxelDataBuffer, CacheEntry cacheEntry, float3 value, uint sampleData)
{
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY)
        return;

    uint3 scaledRadiance = value * SHARC_RADIANCE_SCALE;

    uint prevValue;
    uint baseAddress = cacheEntry * 16;
    if (scaledRadiance.x) voxelDataBuffer.InterlockedAdd(baseAddress + 0, scaledRadiance.x, prevValue);
    if (scaledRadiance.y) voxelDataBuffer.InterlockedAdd(baseAddress + 4, scaledRadiance.y, prevValue);
    if (scaledRadiance.z) voxelDataBuffer.InterlockedAdd(baseAddress + 8, scaledRadiance.z, prevValue);
    if (sampleData) voxelDataBuffer.InterlockedAdd(baseAddress + 12, sampleData, prevValue);
}

struct SharcState
{
    GridParameters gridParameters;
    HashMapData hashMapData;

#if SHARC_UPDATE
    CacheEntry cacheEntry[SHARC_PROPOGATION_DEPTH];
    float3 sampleWeight[SHARC_PROPOGATION_DEPTH];
    uint pathLength;
#endif // SHARC_UPDATE

    RWByteAddressBuffer voxelDataBuffer;

#if SHARC_ENABLE_CACHE_RESAMPLING
    RWByteAddressBuffer voxelDataBufferPrev;
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

void SharcInit(inout SharcState sharcState)
{
#if SHARC_UPDATE
    sharcState.pathLength = 0;
#endif // SHARC_UPDATE
}

void SharcUpdateMiss(inout SharcState sharcState, float3 radiance)
{
#if SHARC_UPDATE
    [loop]
    for (int i = 0; i < sharcState.pathLength; ++i)
    {
        radiance *= sharcState.sampleWeight[i];
        SharcAddVoxelData(sharcState.voxelDataBuffer, sharcState.cacheEntry[i], radiance, 0);
    }
#endif // SHARC_UPDATE
}

bool SharcUpdateHit(inout SharcState sharcState, SharcHitData sharcHitData, float3 lighting, float random)
{
    bool continueTracing = true;
#if SHARC_UPDATE
    uint i = 0;
    [loop]
    for (i = sharcState.pathLength; i > 0; --i)
    {
        sharcState.cacheEntry[i] = sharcState.cacheEntry[i - 1];
        sharcState.sampleWeight[i] = sharcState.sampleWeight[i - 1];
    }

    sharcState.cacheEntry[0] = sharcState.hashMapData.InsertEntry(sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);

    float3 sharcRadiance = lighting;

#if SHARC_ENABLE_CACHE_RESAMPLING && (SHARC_PROPOGATION_DEPTH > 1)
    uint resamplingDepth = uint(lerp(SHARC_RESAMPLING_DEPTH_MIN, SHARC_PROPOGATION_DEPTH, random));
    if (resamplingDepth <= sharcState.pathLength)
    {
        SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBufferPrev, sharcState.cacheEntry[0]);
        if (voxelData.sampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
        {
            sharcRadiance = voxelData.radiance / voxelData.sampleNum;
            continueTracing = false;
        }
    }
#endif // SHARC_ENABLE_CACHE_RESAMPLING

    if (continueTracing)
        SharcAddVoxelData(sharcState.voxelDataBuffer, sharcState.cacheEntry[0], sharcRadiance, 1);

#if SHARC_SEPARATE_EMISSIVE
    sharcRadiance += sharcHitData.emissive;
#endif // SHARC_SEPARATE_EMISSIVE

    sharcState.pathLength += 1;
    sharcState.pathLength = min(sharcState.pathLength, SHARC_PROPOGATION_DEPTH - 1);

    [loop]
    for (i = 1; i < sharcState.pathLength; ++i)
    {
        sharcRadiance *= sharcState.sampleWeight[i];
        SharcAddVoxelData(sharcState.voxelDataBuffer, sharcState.cacheEntry[i], sharcRadiance, 0);
    }
#endif // SHARC_UPDATE
    return continueTracing;
}

void SharcSetThroughput(inout SharcState sharcState, float3 throughput)
{
#if SHARC_UPDATE
    sharcState.sampleWeight[0] = throughput;
#endif // SHARC_UPDATE
}

bool SharcGetCachedRadiance(inout SharcState sharcState, SharcHitData sharcHitData, out float3 radiance, bool debug = false)
{
    if (debug) radiance = 0;
    const uint sampleThreshold = debug ? 0 : SHARC_SAMPLE_NUM_THRESHOLD;

    CacheEntry cacheEntry = sharcState.hashMapData.FindEntry(sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY)
        return false;

    SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBuffer, cacheEntry);
    if (voxelData.sampleNum > sampleThreshold)
    {
        radiance = voxelData.radiance / (float)voxelData.sampleNum;

#if SHARC_SEPARATE_EMISSIVE
        radiance += sharcHitData.emissive;
#endif // SHARC_SEPARATE_EMISSIVE

        return true;
    }

    return false;
}

void SharcCopyHashEntry(uint32_t entryIndex, HashMapData hashMapData, RWStructuredBuffer<uint> copyOffsetBuffer)
{
#if SHARC_DEFERRED_HASH_COMPACTION
    if (entryIndex >= hashMapData.capacity)
        return;

    uint copyOffset = copyOffsetBuffer[entryIndex];
    if (copyOffset == 0)
        return;

    if (copyOffset == HASH_GRID_INVALID_CACHE_ENTRY)
    {
        hashMapData.hashEntriesBuffer[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
    }
    else if (copyOffset)
    {
        HashKey hashKey = hashMapData.hashEntriesBuffer[entryIndex];
        hashMapData.hashEntriesBuffer[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
        hashMapData.hashEntriesBuffer[copyOffset] = hashKey;
    }

    copyOffsetBuffer[entryIndex] = 0;
#endif // SHARC_DEFERRED_HASH_COMPACTION
}

int SharcGetGridDistance2(int3 position)
{
    return position.x * position.x + position.y * position.y + position.z * position.z;
}

HashKey SharcGetAdjacentLevelHashKey(HashKey hashKey, GridParameters gridParameters)
{
    const uint32_t negativeBit = 1 << (HASH_GRID_POSITION_BIT_NUM - 1);
    const uint32_t negativeNumberMask = ~((1 << HASH_GRID_POSITION_BIT_NUM) - 1);

    int3 gridPosition;
    gridPosition.x = int((hashKey >> HASH_GRID_POSITION_BIT_NUM * 0) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.y = int((hashKey >> HASH_GRID_POSITION_BIT_NUM * 1) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.z = int((hashKey >> HASH_GRID_POSITION_BIT_NUM * 2) & HASH_GRID_POSITION_BIT_MASK);

    // Fix negative coordinates
    gridPosition.x = (gridPosition.x & negativeBit) ? gridPosition.x | negativeNumberMask : gridPosition.x;
    gridPosition.y = (gridPosition.y & negativeBit) ? gridPosition.y | negativeNumberMask : gridPosition.y;
    gridPosition.z = (gridPosition.z & negativeBit) ? gridPosition.z | negativeNumberMask : gridPosition.z;

    int level = uint((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 3)) & HASH_GRID_LEVEL_BIT_MASK);

    float voxelSize = GetVoxelSize(level, gridParameters);
    int3 cameraGridPosition = floor((gridParameters.cameraPosition + HASH_GRID_POSITION_OFFSET) / voxelSize);
    int3 cameraVector = cameraGridPosition - gridPosition;
    int cameraDistance = SharcGetGridDistance2(cameraVector);

    int3 cameraGridPositionPrev = floor((gridParameters.cameraPositionPrev + HASH_GRID_POSITION_OFFSET) / voxelSize);
    int3 cameraVectorPrev = cameraGridPositionPrev - gridPosition;
    int cameraDistancePrev = SharcGetGridDistance2(cameraVectorPrev);

    if (cameraDistance < cameraDistancePrev)
    {
        gridPosition = floor(gridPosition / gridParameters.logarithmBase);
        level = min(level + 1, HASH_GRID_LEVEL_BIT_MASK);
    }
    else // this may be inaccurate
    {
        gridPosition = floor(gridPosition * gridParameters.logarithmBase);
        level = max(level - 1, 1);
    }

    HashKey modifiedHashKey = (((uint64_t)gridPosition.x & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0))
        | (((uint64_t)gridPosition.y & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1))
        | (((uint64_t)gridPosition.z & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2))
        | (((uint64_t)level & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

#if HASH_GRID_USE_NORMALS
    modifiedHashKey |= hashKey & ((uint64_t)HASH_GRID_NORMAL_BIT_MASK << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
#endif // HASH_GRID_USE_NORMALS

    return modifiedHashKey;
}

void SharcResolveEntry(uint32_t entryIndex, GridParameters gridParameters, HashMapData hashMapData, RWStructuredBuffer<uint> copyOffsetBuffer,
    RWByteAddressBuffer voxelDataBuffer, RWByteAddressBuffer voxelDataBufferPrev)
{
    if (entryIndex >= hashMapData.capacity)
        return;

    HashKey hashKey = hashMapData.hashEntriesBuffer[entryIndex];
    if (hashKey == HASH_GRID_INVALID_HASH_KEY)
        return;

    static const uint accumulatedSampleNumMax = SHARC_SAMPLE_NUM_MAX;

    uint4 voxelDataPackedPrev = voxelDataBufferPrev.Load4(entryIndex * 16);
    uint4 voxelDataPacked = voxelDataBuffer.Load4(entryIndex * 16);
    uint4 packedData = voxelDataPacked + voxelDataPackedPrev;
    uint sampleNum = packedData.w & SHARC_SAMPLE_COUNTER_BIT_MASK;

#if SHARC_FILTER_ADJACENT_LEVELS
    // Reproject sample from adjacent level
    if ((sampleNum < accumulatedSampleNumMax) && any(gridParameters.cameraPosition.xyz - gridParameters.cameraPositionPrev.xyz) && voxelDataPacked.w)
    {
        HashKey adjacentLevelHashKey = SharcGetAdjacentLevelHashKey(hashKey, gridParameters);

        CacheEntry cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
        if (hashMapData.HashMapFind(adjacentLevelHashKey, cacheEntry))
        {
            uint4 adjacentPackedDataPrev = voxelDataBufferPrev.Load4(cacheEntry * 16);
            uint adjacentSampleNum = adjacentPackedDataPrev.w & SHARC_SAMPLE_COUNTER_BIT_MASK;
            if (adjacentSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
            {
                packedData.xyz += adjacentPackedDataPrev.xyz;
                sampleNum += adjacentSampleNum;
            }
        }
}
#endif // SHARC_FILTER_ADJACENT_LEVELS

    if (sampleNum > accumulatedSampleNumMax)
    {
        packedData.xyz *= (float)accumulatedSampleNumMax / sampleNum;
        sampleNum = accumulatedSampleNumMax;
    }

    uint frameCounter = (voxelDataPackedPrev.w >> SHARC_SAMPLE_COUNTER_BIT_NUM) & SHARC_FRAME_COUNTER_BIT_MASK;
    packedData.w = sampleNum;

    // Increment frame counter for stale samples
    if ((voxelDataPacked.w & SHARC_SAMPLE_COUNTER_BIT_MASK) == 0)
    {
        ++frameCounter;
        packedData.w |= ((frameCounter & SHARC_FRAME_COUNTER_BIT_MASK) << SHARC_SAMPLE_COUNTER_BIT_NUM);
    }

    if (frameCounter > SHARC_STALE_FRAME_NUM_MAX)
    {
        packedData = 0;
#if !SHARC_ENABLE_COMPACTION
        hashMapData.hashEntriesBuffer[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
#endif // !SHARC_ENABLE_COMPACTION
    }

#if SHARC_ENABLE_COMPACTION
    bool isValidElement = packedData.w ? true : false;
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

        voxelDataBuffer.Store4(entryIndex * 16, 0);

        if (isValidElement)
        {
            uint emptySlotIndex = 0;
            while (emptySlotIndex < validElementNum)
            {
                if (((validElementMask >> writeOffset) & 0x1) == 0)
                {
                    if (emptySlotIndex == movableElementIndex)
                    {
                        writeOffset += GetBaseSlot(entryIndex);
#if !SHARC_DEFERRED_HASH_COMPACTION
                        hashMapData.hashEntriesBuffer[writeOffset] = hashKey;
#endif // !SHARC_DEFERRED_HASH_COMPACTION
                        voxelDataBuffer.Store4(writeOffset * 16, packedData);
                        break;
                    }

                    ++emptySlotIndex;
                }

                ++writeOffset;
            }
        }

#if SHARC_DEFERRED_HASH_COMPACTION
        copyOffsetBuffer[entryIndex] = writeOffset ? writeOffset : HASH_GRID_INVALID_CACHE_ENTRY;
#endif // SHARC_DEFERRED_HASH_COMPACTION
    }
    else if (isValidElement)
#endif // SHARC_ENABLE_COMPACTION
    {
        voxelDataBuffer.Store4(entryIndex * 16, packedData);
    }
}
