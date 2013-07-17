//
// Kernel that runs best on Fermi devices
//
// NOTE: compile this .cu module for compute_10,sm_10 with --maxrregcount=124
//

#ifdef WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <cuda.h>

#include "fermi_kernel.h"

#if WIN32
#ifdef _WIN64
#define _64BIT_ALIGN 1
#else
#define _64BIT_ALIGN 0
#endif
#else
#if __x86_64__
#define _64BIT_ALIGN 1
#else
#define _64BIT_ALIGN 0
#endif
#endif

// forward references
template <int WARPS_PER_BLOCK> __global__ void scrypt_core_kernelA(uint32_t *g_idata);
template <int WARPS_PER_BLOCK> __global__ void scrypt_core_kernelB(uint32_t *g_odata);
template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void scrypt_core_kernelB_tex(uint32_t *g_odata);

// scratchbuf constants (pointers to scratch buffer for each work unit)
__constant__ uint32_t* c_V[1024];

// using texture references for the "tex" variants of the B kernels
texture<uint4, 1, cudaReadModeElementType> texRef1D_4_V;
texture<uint4, 2, cudaReadModeElementType> texRef2D_4_V;

FermiKernel::FermiKernel() : KernelInterface()
{
}

bool FermiKernel::bindtexture_1D(uint32_t *d_V, size_t size)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef1D_4_V.normalized = 0;
    texRef1D_4_V.filterMode = cudaFilterModePoint;
    texRef1D_4_V.addressMode[0] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture(NULL, &texRef1D_4_V, d_V, &channelDesc4, size));
    return true;
}

bool FermiKernel::bindtexture_2D(uint32_t *d_V, int width, int height, size_t pitch)
{
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    texRef2D_4_V.normalized = 0;
    texRef2D_4_V.filterMode = cudaFilterModePoint;
    texRef2D_4_V.addressMode[0] = cudaAddressModeClamp;
    texRef2D_4_V.addressMode[1] = cudaAddressModeClamp;
    checkCudaErrors(cudaBindTexture2D(NULL, &texRef2D_4_V, d_V, &channelDesc4, width, height, pitch));
    return true;
}

bool FermiKernel::unbindtexture_1D()
{
    checkCudaErrors(cudaUnbindTexture(texRef1D_4_V));
    return true;
}

bool FermiKernel::unbindtexture_2D()
{
    checkCudaErrors(cudaUnbindTexture(texRef2D_4_V));
    return true;
}

void FermiKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool FermiKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream, uint32_t* d_idata, uint32_t* d_odata, int *mutex, bool interactive, bool benchmark, int texture_cache)
{
    bool success = true;

    // clear CUDA's error variable
    cudaGetLastError();

    // First phase: Sequential writes to scratchpad.

    switch (WARPS_PER_BLOCK) {
        case 1: scrypt_core_kernelA<1><<< grid, threads, 0, stream >>>(d_idata); break;
        case 2: scrypt_core_kernelA<2><<< grid, threads, 0, stream >>>(d_idata); break;
        case 3: scrypt_core_kernelA<3><<< grid, threads, 0, stream >>>(d_idata); break;
        case 4: scrypt_core_kernelA<4><<< grid, threads, 0, stream >>>(d_idata); break;
        case 5: scrypt_core_kernelA<5><<< grid, threads, 0, stream >>>(d_idata); break;
        case 6: scrypt_core_kernelA<6><<< grid, threads, 0, stream >>>(d_idata); break;
        case 7: scrypt_core_kernelA<7><<< grid, threads, 0, stream >>>(d_idata); break;
        case 8: scrypt_core_kernelA<8><<< grid, threads, 0, stream >>>(d_idata); break;
        default: success = false; break;
    }

    // Optional millisecond sleep in between kernels

    if (!benchmark && interactive) {
        checkCudaErrors(MyStreamSynchronize(stream, 1, thr_id));
#ifdef WIN32
        Sleep(1);
#else
        usleep(1000);
#endif
    }

    // Second phase: Random read access from scratchpad.

    if (texture_cache)
    {
        if (texture_cache == 1)
        {
            switch (WARPS_PER_BLOCK) {
                case 1: scrypt_core_kernelB_tex<1,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 2: scrypt_core_kernelB_tex<2,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 3: scrypt_core_kernelB_tex<3,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 4: scrypt_core_kernelB_tex<4,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 5: scrypt_core_kernelB_tex<5,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 6: scrypt_core_kernelB_tex<6,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 7: scrypt_core_kernelB_tex<7,1><<< grid, threads, 0, stream >>>(d_odata); break;
                case 8: scrypt_core_kernelB_tex<8,1><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        }
        else if (texture_cache == 2)
        {
            switch (WARPS_PER_BLOCK) {
                case 1: scrypt_core_kernelB_tex<1,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 2: scrypt_core_kernelB_tex<2,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 3: scrypt_core_kernelB_tex<3,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 4: scrypt_core_kernelB_tex<4,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 5: scrypt_core_kernelB_tex<5,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 6: scrypt_core_kernelB_tex<6,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 7: scrypt_core_kernelB_tex<7,2><<< grid, threads, 0, stream >>>(d_odata); break;
                case 8: scrypt_core_kernelB_tex<8,2><<< grid, threads, 0, stream >>>(d_odata); break;
                default: success = false; break;
            }
        } else success = false;
    }
    else
    {
        switch (WARPS_PER_BLOCK) {
            case 1: scrypt_core_kernelB<1><<< grid, threads, 0, stream >>>(d_odata); break;
            case 2: scrypt_core_kernelB<2><<< grid, threads, 0, stream >>>(d_odata); break;
            case 3: scrypt_core_kernelB<3><<< grid, threads, 0, stream >>>(d_odata); break;
            case 4: scrypt_core_kernelB<4><<< grid, threads, 0, stream >>>(d_odata); break;
            case 5: scrypt_core_kernelB<5><<< grid, threads, 0, stream >>>(d_odata); break;
            case 6: scrypt_core_kernelB<6><<< grid, threads, 0, stream >>>(d_odata); break;
            case 7: scrypt_core_kernelB<7><<< grid, threads, 0, stream >>>(d_odata); break;
            case 8: scrypt_core_kernelB<8><<< grid, threads, 0, stream >>>(d_odata); break;
            default: success = false; break;
        }
    }

    // catch any kernel launch failures
    if (cudaPeekAtLastError() != cudaSuccess) success = false;

    return success;
}

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

static __host__ __device__ void xor_salsa8(uint32_t * const B, const uint32_t * const C)
{
    uint32_t x0 = (B[ 0] ^= C[ 0]), x1 = (B[ 1] ^= C[ 1]), x2 = (B[ 2] ^= C[ 2]), x3 = (B[ 3] ^= C[ 3]);
    uint32_t x4 = (B[ 4] ^= C[ 4]), x5 = (B[ 5] ^= C[ 5]), x6 = (B[ 6] ^= C[ 6]), x7 = (B[ 7] ^= C[ 7]);
    uint32_t x8 = (B[ 8] ^= C[ 8]), x9 = (B[ 9] ^= C[ 9]), xa = (B[10] ^= C[10]), xb = (B[11] ^= C[11]);
    uint32_t xc = (B[12] ^= C[12]), xd = (B[13] ^= C[13]), xe = (B[14] ^= C[14]), xf = (B[15] ^= C[15]);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);
        
    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    /* Operate on columns. */
    x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
    x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
    xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
    x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);
        
    /* Operate on rows. */
    x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
    x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
    x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
    x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

    B[ 0] += x0; B[ 1] += x1; B[ 2] += x2; B[ 3] += x3; B[ 4] += x4; B[ 5] += x5; B[ 6] += x6; B[ 7] += x7;
    B[ 8] += x8; B[ 9] += x9; B[10] += xa; B[11] += xb; B[12] += xc; B[13] += xd; B[14] += xe; B[15] += xf;
}

static __host__ __device__ uint4& operator^=(uint4& left, const uint4& right)
{
    left.x ^= right.x;
    left.y ^= right.y;
    left.z ^= right.z;
    left.w ^= right.w;
    return left;
}

////////////////////////////////////////////////////////////////////////////////
//! Scrypt core kernel
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernelA(uint32_t *g_idata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx        = threadIdx.x / warpSize;
    volatile int warpThread     = threadIdx.x % warpSize;

    // add block specific offsets
    volatile int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_idata += 32 * offset;
    uint32_t * volatile V = c_V[offset/WU_PER_WARP];

    // variables supporting the large memory transaction magic
    volatile unsigned int Y = warpThread/4;
    volatile unsigned int Z = 4*(warpThread%4);

    // registers to store an entire work unit
    uint32_t B[16], C[16];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&V[SCRATCH*(wu+Y)+Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z])) = *((uint4*)(&g_idata[32*(wu+Y)+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&V[SCRATCH*(wu+Y)+16+Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z])) = *((uint4*)(&g_idata[32*(wu+Y)+16+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

    for (int i = 1; i < 1024; i++) {

        xor_salsa8(B, C); xor_salsa8(C, B);

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&V[SCRATCH*(wu+Y) + i*32 + Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z]));

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&V[SCRATCH*(wu+Y) + i*32 + 16 + Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z]));
    }
}

template <int WARPS_PER_BLOCK> __global__ void
scrypt_core_kernelB(uint32_t *g_odata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx        = threadIdx.x / warpSize;
    volatile int warpThread     = threadIdx.x % warpSize;

    // add block specific offsets
    volatile int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset;
    uint32_t * volatile V = c_V[offset/WU_PER_WARP];

    // variables supporting the large memory transaction magic
    volatile unsigned int Y = warpThread/4;
    volatile unsigned int Z = 4*(warpThread%4);

    // registers to store an entire work unit
    uint32_t B[16], C[16];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&X[warpIdx][wu+Y][Z])) = *((uint4*)(&V[SCRATCH*(wu+Y) + 1023*32 + Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&X[warpIdx][wu+Y][Z])) = *((uint4*)(&V[SCRATCH*(wu+Y) + 1023*32 + 16+Z]));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

    xor_salsa8(B, C); xor_salsa8(C, B);

    for (int i = 0; i < 1024; i++) {

        X[warpIdx][warpThread][16] = C[0];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&X[warpIdx][wu+Y][Z])) ^= *((uint4*)(&V[SCRATCH*(wu+Y) + 32*(X[warpIdx][wu+Y][16] & 1023) + Z]));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&X[warpIdx][wu+Y][Z])) ^= *((uint4*)(&V[SCRATCH*(wu+Y) + 32*(X[warpIdx][wu+Y][16] & 1023) + 16 + Z]));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

        xor_salsa8(B, C); xor_salsa8(C, B);
    }

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z]));

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+16+Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z]));
}

template <int WARPS_PER_BLOCK, int TEX_DIM> __global__ void
scrypt_core_kernelB_tex(uint32_t *g_odata)
{
    __shared__ uint32_t X[WARPS_PER_BLOCK][WU_PER_WARP][16+1+_64BIT_ALIGN]; // +1 to resolve bank conflicts

    volatile int warpIdx        = threadIdx.x / warpSize;
    volatile int warpThread     = threadIdx.x % warpSize;

    // add block specific offsets
    volatile int offset = blockIdx.x * WU_PER_BLOCK + warpIdx * WU_PER_WARP;
    g_odata += 32 * offset;

    // variables supporting the large memory transaction magic
    volatile unsigned int Y = warpThread/4;
    volatile unsigned int Z = 4*(warpThread%4);

    // registers to store an entire work unit
    uint32_t B[16], C[16];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&X[warpIdx][wu+Y][Z])) = ((TEX_DIM == 1) ?
                    tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 1023*32 + Z)/4) :
                    tex2D(texRef2D_4_V, 0.5f + (32*1023 + Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&X[warpIdx][wu+Y][Z])) = ((TEX_DIM == 1) ?
                    tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 1023*32 + 16+Z)/4) :
                    tex2D(texRef2D_4_V, 0.5f + (32*1023 + 16+Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
    for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

    xor_salsa8(B, C); xor_salsa8(C, B);

    for (int i = 0; i < 1024; i++) {

        X[warpIdx][warpThread][16] = C[0];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&X[warpIdx][wu+Y][Z])) ^= ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 32*(X[warpIdx][wu+Y][16] & 1023) + Z)/4) :
                        tex2D(texRef2D_4_V, 0.5f + (32*(X[warpIdx][wu+Y][16] & 1023) + Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) B[idx] = X[warpIdx][warpThread][idx];

#pragma unroll 16
        for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 4
        for (int wu=0; wu < 32; wu+=8)
            *((uint4*)(&X[warpIdx][wu+Y][Z])) ^= ((TEX_DIM == 1) ?
                        tex1Dfetch(texRef1D_4_V, (SCRATCH*(offset+wu+Y) + 32*(X[warpIdx][wu+Y][16] & 1023) + 16+Z)/4) :
                        tex2D(texRef2D_4_V, 0.5f + (32*(X[warpIdx][wu+Y][16] & 1023) + 16+Z)/4, 0.5f + (offset+wu+Y)));
#pragma unroll 16
        for (int idx=0; idx < 16; idx++) C[idx] = X[warpIdx][warpThread][idx];

        xor_salsa8(B, C); xor_salsa8(C, B);
    }

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = B[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z]));

#pragma unroll 16
    for (int idx=0; idx < 16; ++idx) X[warpIdx][warpThread][idx] = C[idx];
#pragma unroll 4
    for (int wu=0; wu < 32; wu+=8)
        *((uint4*)(&g_odata[32*(wu+Y)+16+Z])) = *((uint4*)(&X[warpIdx][wu+Y][Z]));
}
