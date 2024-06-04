#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define H_RES 3840 // horizontal resolution
#define V_RES 2160 // vertical resolution
#define CENTER_X 0 // X coordinate for image center
#define CENTER_Y 0 // Y coordinate for image center
#define SCALE 2 // maximum X value in the fractal graph
#define ITERATIONS (1 << 8) // number of iteration for checking divergence
#define R (1 << 8) // ceiling upon which function is considered divergent
#define SHADOW_DISTANCE 16 // radius of the circular shadow plot
#define SHADOW_SHARPNESS 1 // rapidity with which shadow gets dark
#define SHADOW_TILT_H -64 // horizontal offset from where shadow is plotted
#define SHADOW_TILT_V 32 // vertical offset from where shadow is plotted
#define SHADOW_INTENSITY 0.8 // blackness of the shadow

typedef unsigned char byte;
typedef struct complex {
    double r;
    double i;
} complex;

/// Calculate fractal and shadow for each pixel point and assign images respectively
__host__ void generate_fractal(const complex *c, byte *image, const byte *inside, const byte *outside);

/// Complex multiplication
__device__ void cmul(complex *outcome, const complex *first, const complex *second);

/// Complex sum
__device__ void csum(complex *outcome, const complex *first, const complex *second);

/// Complex absolute value
__device__ double cmod(const complex *z);

/// Save ppm image on disk
int save_image(const char *filename, unsigned char *image);

/// Load ppm image from disk
int load_image(const char *filename, unsigned char *image);

/// Measure milliseconds
double milliseconds();

int main(int argc, char **argv) {

    // Retrieve c constant from input args
    if (argc != 3) {
        printf("Provide a complex number");
        return 1;
    }
    complex c;
    c.r = strtod(argv[1], NULL);
    c.i = strtod(argv[2], NULL);

    // Allocate memory for input and output images
    byte *image = (byte *)malloc(3 * V_RES * H_RES * sizeof(byte));
    byte *inside = (byte *)malloc(3 * V_RES * H_RES * sizeof(byte));
    byte *outside = (byte *)malloc(3 * V_RES * H_RES * sizeof(byte));

    // Load the two input images
    if (load_image("inside.ppm", inside) < 0) {
        fprintf(stderr, "Error opening %s\n", "inside.ppm");
        return 1;
    }
    if (load_image("outside.ppm", outside) < 0) {
        fprintf(stderr, "Error opening %s\n", "outside.ppm");
        return 1;
    }

    // Compute fractal, shadow and image assignment
    generate_fractal(&c, image, inside, outside);

    // Save the output image
    if (save_image("fractal.ppm", image) < 0) {
        fprintf(stderr, "Error opening %s\n", "fractal.ppm");
        return 1;
    }

    // Free images memory
    free(image);
    free(inside);
    free(outside);
    return 0;
}

#define OUT 0xFF // outside color of the fractal mask
#define IN 0x00 // inside color of the fractal mask

/// The first iteration generates a black and white image (mask) where
/// pixels inside the fractal are black and pixels outside are white.
__global__ void __compute_mask(
    int h_max,
    int v_max,
    const complex c,
    byte *__restrict__ mask);

/// The second iteration plots a circular shadow for each white pixel of
/// the just generated fractal mask.
/// This is done by adding one to the corresponding elements of the shadow
/// integer array (sized like fractal mask).
/// The higher is the number, the higher is the shadow intensity.
__global__ void __apply_shadow(
    int h_max,
    int v_max,
    const byte *__restrict__ mask,
    int *__restrict__ shadow);

/// The third iteration assigns the inside and the outside images.
/// The shadow toner for the inner image is computed starting from the corresponding
/// value in the shadow array.
__global__ void __assign_final(
    const int *__restrict__ shadow,
    const byte *__restrict__ mask,
    const byte *__restrict__ inside,
    const byte *__restrict__ outside,
    byte *__restrict__ image);

__host__ void generate_fractal(const complex *c, byte *image, const byte *inside, const byte *outside) {
    // Block dimensions for kernels
#define BLOCK_DIM 16
    // Required vertical dimension due to shadow offset
#define H_EXTENDED (H_RES + 2 * (abs(SHADOW_TILT_H) + SHADOW_DISTANCE))
    // Required horizontal dimension due to shadow offset
#define V_EXTENDED (V_RES + 2 * (abs(SHADOW_TILT_V) + SHADOW_DISTANCE))

    // Initialize events
    float time;
    double start_ms, stop_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory
    byte *mask_d, *inside_d, *outside_d, *image_d;
    int *shadow_d;
    cudaMalloc(&mask_d, V_EXTENDED * H_EXTENDED * sizeof(byte));
    cudaMalloc(&shadow_d, V_EXTENDED * H_EXTENDED * sizeof(int));
    cudaMalloc(&inside_d, 3 * V_RES * H_RES * sizeof(byte));
    cudaMalloc(&outside_d, 3 * V_RES * H_RES * sizeof(byte));
    cudaMalloc(&image_d, 3 * V_RES * H_RES * sizeof(byte));

    // Data initialization
    cudaMemset(mask_d, 0x00, V_EXTENDED * H_EXTENDED * sizeof(byte));
    cudaMemset(shadow_d, 0, V_EXTENDED * H_EXTENDED * sizeof(int));

    // Data transfer to device
    start_ms = milliseconds();
    cudaMemcpy(inside_d, inside, 3 * V_RES * H_RES * sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(outside_d, outside, 3 * V_RES * H_RES * sizeof(byte), cudaMemcpyHostToDevice);
    stop_ms = milliseconds();
    printf("Memory transfer to device: %f\n", stop_ms - start_ms);

    dim3 block_size, grid_size;
    block_size = dim3(BLOCK_DIM, BLOCK_DIM);

    // For each pixel compute fractal mask
    cudaEventRecord(start);
    grid_size = dim3(ceil(H_EXTENDED / block_size.x), ceil(V_EXTENDED / block_size.y));
    __compute_mask << <grid_size, block_size >> > (H_EXTENDED, V_EXTENDED, *c, mask_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    printf("Mask computation: %f\n", time);

    // For each pixel compute shadow value
    cudaEventRecord(start);
    for (int i = 0; i < 4; i++) {
        grid_size = dim3(
            (ceil(H_EXTENDED / block_size.x) + i % 2) / 2,
            (ceil(V_EXTENDED / block_size.y) + i / 2 % 2) / 2);
        __apply_shadow << <grid_size, block_size >> > (i, mask_d, shadow_d);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    printf("Shadow application: %f\n", time);

    // For each pixel select final image, computing its shadow
    cudaEventRecord(start);
    grid_size = dim3(ceil(H_RES / block_size.x), ceil(V_RES / block_size.y));
    __assign_final << <grid_size, block_size >> > (shadow_d, mask_d, inside_d, outside_d, image_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    printf("Final assignment: %f\n", time);

    // Data transfer to host
    start_ms = milliseconds();
    cudaMemcpy(image, image_d, 3 * V_RES * H_RES * sizeof(byte), cudaMemcpyDeviceToHost);
    stop_ms = milliseconds();
    printf("Memory transfer to host: %f\n", stop_ms - start_ms);

    // Free the allocated memory
    cudaFree(mask_d);
    cudaFree(shadow_d);
    cudaFree(inside_d);
    cudaFree(outside_d);
    cudaFree(image_d);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

#undef H_EXTENDED
#undef V_EXTENDED
#undef BLOCK_DIM
}

__global__ void __compute_mask(
    int h_max,
    int v_max,
    const complex c,
    byte *__restrict__ mask) {

    // Calculate index of the pixel
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= h_max || v >= v_max) return;
    int idx = v * h_max + h;

    // Calculate the side length of a pixel in the complex plane
    double res_unit = (double)SCALE / (H_RES / 2);

    // Calculate coordinates of the pixel in the complex plane
    complex z0, z1;
    z0.r = res_unit * (h - h_max / 2) + CENTER_X;
    z0.i = res_unit * (v - v_max / 2) + CENTER_Y;

    // Iterate the function on itself to then analyze the convergence
    for (int i = 0; i < ITERATIONS; i++) {

        // Compute function z1 = z0^2 + c
        cmul(&z1, &z0, &z0);
        csum(&z1, &z1, &c);
        z0.r = z1.r;
        z0.i = z1.i;

        // Check if function has diverged
        if (cmod(&z0) > R) {

            // Assign outside value
            mask[idx] = OUT;
            return;
        }
    }
}

__global__ void __apply_shadow(
    int h_max,
    int v_max,
    __restrict__ const byte *mask,
    int *__restrict__ shadow) {

    // Calculate index of the pixel
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= h_max || v >= v_max) return;
    int idx = v * h_max + h;

    // Ignore points in the image below
    if (mask[idx] == IN) return;

    // Plot a shadow circle
#pragma unroll
    for (int i = -SHADOW_DISTANCE; i < SHADOW_DISTANCE; i++) {
#pragma unroll
        for (int j = -SHADOW_DISTANCE; j < SHADOW_DISTANCE; j++) {
            __syncthreads();

            // Calculate index of the offset shadow
            int shadow_idx = (v + j + SHADOW_TILT_V) * h_max + h + i + SHADOW_TILT_H;

            // Check that the current shadow index is inside borders and radius
            if (shadow_idx < 0 || shadow_idx >= h_max * v_max ||
                sqrt(pow(idx % h_max - shadow_idx % h_max + SHADOW_TILT_H, 2) +
                    pow(idx / h_max - shadow_idx / h_max + SHADOW_TILT_V, 2)) > SHADOW_DISTANCE)
                continue;

            atomicAdd(&shadow[shadow_idx], 1);
        }
    }
}

__global__ void __assign_final(
    const int *__restrict__ shadow,
    const byte *__restrict__ mask,
    const byte *__restrict__ inside,
    const byte *__restrict__ outside,
    byte *__restrict__ image) {

    // Calculate index of the pixel
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= H_RES || v >= V_RES) return;
    int idx = v * H_RES + h;
    int idx_framed = (abs(SHADOW_TILT_V) + SHADOW_DISTANCE + v) *
        (H_RES + 2 * (abs(SHADOW_TILT_H) + SHADOW_DISTANCE)) +
        abs(SHADOW_TILT_H) + SHADOW_DISTANCE + h;

    if (mask[idx_framed] == OUT) {

        // Outside image assignment
        image[3 * idx + 0] = outside[3 * idx + 0];
        image[3 * idx + 1] = outside[3 * idx + 1];
        image[3 * idx + 2] = outside[3 * idx + 2];

    } else {

        // Shadow intensity computation
        float toner = (exp(-SHADOW_SHARPNESS * shadow[idx_framed] /
            (3.1416 * SHADOW_DISTANCE * SHADOW_DISTANCE))) *
            SHADOW_INTENSITY + (1 - SHADOW_INTENSITY);

        // Inside image assignment with shadow
        image[3 * idx + 0] = inside[3 * idx + 0] * toner;
        image[3 * idx + 1] = inside[3 * idx + 1] * toner;
        image[3 * idx + 2] = inside[3 * idx + 2] * toner;
    }
}

#undef IN
#undef OUT

__device__ inline void cmul(complex *outcome, const complex *first, const complex *second) {
    outcome->r = first->r * second->r - first->i * second->i;
    outcome->i = first->r * second->i + first->i * second->r;
}

__device__ inline void csum(complex *outcome, const complex *first, const complex *second) {
    outcome->r = first->r + second->r;
    outcome->i = first->i + second->i;
}

__device__ inline double cmod(const complex *z) {
    return sqrt(z->r * z->r + z->i * z->i);
}

int save_image(const char *filename, unsigned char *image) {
    FILE *f = fopen(filename, "wb");
    if (f == NULL) return -1;
    fprintf(f, "P6\n%d %d\n%d\n", H_RES, V_RES, 255);
    fwrite(image, sizeof(unsigned char), H_RES * V_RES * 3, f);
    fclose(f);
    return 0;
}

int load_image(const char *filename, unsigned char *image) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL) return -1;
    char temp1[4];
    int temp2, h, v;
    fscanf(f, "%s\n%d %d\n%d\n", temp1, &h, &v, &temp2);
    if (h != H_RES || v != V_RES) {
        fclose(f);
        return -1;
    }
    fread(image, sizeof(unsigned char), H_RES * V_RES * 3, f);
    fclose(f);
    return 0;
}

inline double milliseconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec * 0.001;
}
