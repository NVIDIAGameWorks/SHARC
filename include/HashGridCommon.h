/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define HASH_GRID_POSITION_BIT_NUM          17
#define HASH_GRID_POSITION_BIT_MASK         ((1u << HASH_GRID_POSITION_BIT_NUM) - 1)
#define HASH_GRID_LEVEL_BIT_NUM             10
#define HASH_GRID_LEVEL_BIT_MASK            ((1u << HASH_GRID_LEVEL_BIT_NUM) - 1)
#define HASH_GRID_NORMAL_BIT_NUM            3
#define HASH_GRID_NORMAL_BIT_MASK           ((1u << HASH_GRID_NORMAL_BIT_NUM) - 1)
#define HASH_GRID_HASH_MAP_BUCKET_SIZE      32
#define HASH_GRID_INVALID_HASH_KEY          0
#define HASH_GRID_INVALID_CACHE_ENTRY       0xFFFFFFFF
#define HASH_GRID_USE_NORMALS               1       // account for normal data in the hash key
#define HASH_GRID_ALLOW_COMPACTION          (HASH_GRID_HASH_MAP_BUCKET_SIZE == 32)
#define HASH_GRID_LEVEL_BIAS                2       // positive bias adds extra levels with content magnification (can be negative as well)
#define HASH_GRID_POSITION_OFFSET           float3(0.0f, 0.0f, 0.0f)
#define HASH_GRID_POSITION_BIAS             1e-4f   // may require adjustment for extreme scene scales
#define HASH_GRID_NORMAL_BIAS               1e-3f

#define CacheEntry uint
#define HashKey uint64_t

struct GridParameters
{
    float3 cameraPosition;
    float3 cameraPositionPrev;
    float logarithmBase;
    float sceneScale;
};

float LogBase(float x, float base)
{
    return log(x) / log(base);
}

uint GetBaseSlot(uint slot, uint capacity)
{
#if HASH_GRID_ALLOW_COMPACTION
    return (slot / HASH_GRID_HASH_MAP_BUCKET_SIZE) * HASH_GRID_HASH_MAP_BUCKET_SIZE;
#else // !HASH_GRID_ALLOW_COMPACTION
    return min(slot, capacity - HASH_GRID_HASH_MAP_BUCKET_SIZE);
#endif // !HASH_GRID_ALLOW_COMPACTION
}

// http://burtleburtle.net/bob/hash/integer.html
uint HashJenkins32(uint a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

uint Hash32(HashKey hashKey)
{
    return HashJenkins32(uint((hashKey >> 0) & 0xffffffff))
         ^ HashJenkins32(uint((hashKey >> 32) & 0xffffffff));
}

uint GetGridLevel(float3 samplePosition, GridParameters gridParameters)
{
    const float distance2 = dot(gridParameters.cameraPosition - samplePosition, gridParameters.cameraPosition - samplePosition);

    return uint(clamp(0.5f * LogBase(distance2, gridParameters.logarithmBase) + HASH_GRID_LEVEL_BIAS, 1.0f, float(HASH_GRID_LEVEL_BIT_MASK)));
}

float GetVoxelSize(uint gridLevel, GridParameters gridParameters)
{
    return pow(gridParameters.logarithmBase, gridLevel) / (gridParameters.sceneScale * pow(gridParameters.logarithmBase, HASH_GRID_LEVEL_BIAS));
}

// Based on logarithmic caching by Johannes Jendersie
int4 CalculateGridPositionLog(float3 samplePosition, GridParameters gridParameters)
{
    samplePosition += float3(HASH_GRID_POSITION_BIAS, HASH_GRID_POSITION_BIAS, HASH_GRID_POSITION_BIAS);

    uint  gridLevel    = GetGridLevel(samplePosition, gridParameters);
    float voxelSize    = GetVoxelSize(gridLevel, gridParameters);
    int3  gridPosition = int3(floor(samplePosition / voxelSize));

    return int4(gridPosition.xyz, gridLevel);
}

HashKey ComputeSpatialHash(float3 samplePosition, float3 sampleNormal, GridParameters gridParameters)
{
    uint4 gridPosition = uint4(CalculateGridPositionLog(samplePosition, gridParameters));

    HashKey hashKey = ((uint64_t(gridPosition.x) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0))
                    | ((uint64_t(gridPosition.y) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1))
                    | ((uint64_t(gridPosition.z) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2))
                    | ((uint64_t(gridPosition.w) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

#if HASH_GRID_USE_NORMALS
    uint normalBits =
        (sampleNormal.x + HASH_GRID_NORMAL_BIAS >= 0 ? 1 : 0) +
        (sampleNormal.y + HASH_GRID_NORMAL_BIAS >= 0 ? 2 : 0) +
        (sampleNormal.z + HASH_GRID_NORMAL_BIAS >= 0 ? 4 : 0);

    hashKey |= (uint64_t(normalBits) << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
#endif // HASH_GRID_USE_NORMALS

    return hashKey;
}

float3 GetPositionFromHashKey(const HashKey hashKey, GridParameters gridParameters)
{
    const int signBit      = 1 << (HASH_GRID_POSITION_BIT_NUM - 1);
    const int signMask     = ~((1 << HASH_GRID_POSITION_BIT_NUM) - 1);

    int3 gridPosition;
    gridPosition.x = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 0)) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.y = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 1)) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.z = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 2)) & HASH_GRID_POSITION_BIT_MASK);

    // Fix negative coordinates
    gridPosition.x = (gridPosition.x & signBit) != 0 ? gridPosition.x | signMask : gridPosition.x;
    gridPosition.y = (gridPosition.y & signBit) != 0 ? gridPosition.y | signMask : gridPosition.y;
    gridPosition.z = (gridPosition.z & signBit) != 0 ? gridPosition.z | signMask : gridPosition.z;

    uint   gridLevel      = uint((hashKey >> HASH_GRID_POSITION_BIT_NUM * 3) & HASH_GRID_LEVEL_BIT_MASK);
    float  voxelSize      = GetVoxelSize(gridLevel, gridParameters);
    float3 samplePosition = (gridPosition + 0.5f) * voxelSize;

    return samplePosition;
}

struct HashMapData
{
    uint capacity;

    RW_STRUCTURED_BUFFER(hashEntriesBuffer, uint64_t);

#if !HASH_GRID_ENABLE_64_BIT_ATOMICS
    RW_STRUCTURED_BUFFER(lockBuffer, uint);
#endif // !HASH_GRID_ENABLE_64_BIT_ATOMICS
};

void HashMapAtomicCompareExchange(in HashMapData hashMapData, in uint dstOffset, in uint64_t compareValue, in uint64_t value, out uint64_t originalValue)
{
#if HASH_GRID_ENABLE_64_BIT_ATOMICS
#if SHARC_ENABLE_GLSL
    originalValue = InterlockedCompareExchange(BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset), compareValue, value);
#else // !SHARC_ENABLE_GLSL
    InterlockedCompareExchange(BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset), compareValue, value, originalValue);
#endif // !SHARC_ENABLE_GLSL
#else // !HASH_GRID_ENABLE_64_BIT_ATOMICS
    // ANY rearangments to the code below lead to device hang if fuse is unlimited
    const uint cLock = 0xAAAAAAAA;
    uint fuse = 0;
    const uint fuseLength = 8;
    bool busy = true;
    while (busy && fuse < fuseLength)
    {
        uint state;
        InterlockedExchange(hashMapData.lockBuffer[dstOffset], cLock, state);
        busy = state != 0;

        if (state != cLock)
        {
            originalValue = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset);
            if (originalValue == compareValue)
                BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset) = value;
            InterlockedExchange(hashMapData.lockBuffer[dstOffset], state, fuse);
            fuse = fuseLength;
        }
        ++fuse;
    }
#endif // !HASH_GRID_ENABLE_64_BIT_ATOMICS
}

bool HashMapInsert(in HashMapData hashMapData, const HashKey hashKey, out CacheEntry cacheEntry)
{
    uint    hash = Hash32(hashKey);
    uint    slot = hash % hashMapData.capacity;
    uint    initSlot = slot;
    HashKey prevHashKey = HASH_GRID_INVALID_HASH_KEY;

    const uint baseSlot = GetBaseSlot(slot, hashMapData.capacity);
    for (uint bucketOffset = 0; bucketOffset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucketOffset)
    {
        HashMapAtomicCompareExchange(hashMapData, baseSlot + bucketOffset, HASH_GRID_INVALID_HASH_KEY, hashKey, prevHashKey);

        if (prevHashKey == HASH_GRID_INVALID_HASH_KEY || prevHashKey == hashKey)
        {
            cacheEntry = baseSlot + bucketOffset;
            return true;
        }
    }

    cacheEntry = 0;
    return false;
}

bool HashMapFind(in HashMapData hashMapData, const HashKey hashKey, inout CacheEntry cacheEntry)
{
    uint    hash = Hash32(hashKey);
    uint    slot = hash % hashMapData.capacity;

    const uint baseSlot = GetBaseSlot(slot, hashMapData.capacity);
    for (uint bucketOffset = 0; bucketOffset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucketOffset)
    {
        HashKey storedHashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, baseSlot + bucketOffset);

        if (storedHashKey == hashKey)
        {
            cacheEntry = baseSlot + bucketOffset;
            return true;
        }
#if HASH_GRID_ALLOW_COMPACTION
        else if (storedHashKey == HASH_GRID_INVALID_HASH_KEY)
        {
            return false;
        }
#endif // HASH_GRID_ALLOW_COMPACTION
    }

    return false;
}

CacheEntry HashMapInsertEntry(in HashMapData hashMapData, float3 samplePosition, float3 sampleNormal, GridParameters gridParameters)
{
    CacheEntry    cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
    const HashKey hashKey    = ComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    bool          successful = HashMapInsert(hashMapData, hashKey, cacheEntry);

    return cacheEntry;
}

CacheEntry HashMapFindEntry(in HashMapData hashMapData, float3 samplePosition, float3 sampleNormal, GridParameters gridParameters)
{
    CacheEntry    cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
    const HashKey hashKey    = ComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    bool          successful = HashMapFind(hashMapData, hashKey, cacheEntry);

    return cacheEntry;
}

// Debug functions
float3 GetColorFromHash32(uint hash)
{
    float3 color;
    color.x = ((hash >>  0) & 0x3ff) / 1023.0f;
    color.y = ((hash >> 11) & 0x7ff) / 2047.0f;
    color.z = ((hash >> 22) & 0x7ff) / 2047.0f;

    return color;
}

// Debug visualization
float3 HashGridDebugColoredHash(float3 samplePosition, GridParameters gridParameters)
{
    HashKey hashKey = ComputeSpatialHash(samplePosition, float3(0, 0, 0), gridParameters);

    uint gridLevel = GetGridLevel(samplePosition, gridParameters);
    float3 color = GetColorFromHash32(Hash32(hashKey)) * GetColorFromHash32(HashJenkins32(gridLevel)).xyz;

    return color;
}

float3 HashGridDebugOccupancy(uint2 pixelPosition, uint2 screenSize, HashMapData hashMapData)
{
    const uint elementSize = 7;
    const uint borderSize = 1;
    const uint blockSize = elementSize + borderSize;

    uint rowNum = screenSize.y / blockSize;
    uint rowIndex = pixelPosition.y / blockSize;
    uint columnIndex = pixelPosition.x / blockSize;
    uint elementIndex = (columnIndex / HASH_GRID_HASH_MAP_BUCKET_SIZE) * (rowNum * HASH_GRID_HASH_MAP_BUCKET_SIZE) + rowIndex * HASH_GRID_HASH_MAP_BUCKET_SIZE + (columnIndex % HASH_GRID_HASH_MAP_BUCKET_SIZE);

    if (elementIndex < hashMapData.capacity && ((pixelPosition.x % blockSize) < elementSize && (pixelPosition.y % blockSize) < elementSize))
    {
        HashKey storedHashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, elementIndex);

        if (storedHashKey != HASH_GRID_INVALID_HASH_KEY)
            return float3(0.0f, 1.0f, 0.0f);
    }

    return float3(0.0f, 0.0f, 0.0f);
}
