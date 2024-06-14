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
#define SHADOW_DISTANCE 64 // radius of the circular shadow plot
#define SHADOW_SHARPNESS 1 // rapidity with which shadow gets dark
#define SHADOW_TILT_H -64 // horizontal offset from where shadow is plotted
#define SHADOW_TILT_V 32 // vertical offset from where shadow is plotted
#define SHADOW_INTENSITY 0.8 // blackness of the shadow
#define BLOCK_DIM 16 // threads per block dimension

typedef unsigned char byte;
typedef struct complex {
    double r;
    double i;
} complex;

/// Check the last kernel call for errors
#define CHECK_KERNELCALL {                                  \
    const cudaError_t err = cudaGetLastError();             \
    if (err != cudaSuccess) {                               \
        printf("ERROR: %s::%d\n -> %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}

/// Calculate fractal and shadow for each pixel point and assign images respectively
__host__ void generate_art(const complex *c, byte *image, const byte *inside, const byte *outside);

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

/// Measure milliseconds on host
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

    // Allocate memory for images
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
    generate_art(&c, image, inside, outside);

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
#define H_EXTENSION (abs(SHADOW_TILT_H) + SHADOW_DISTANCE) // horizontal extension due to shadow offset
#define V_EXTENSION (abs(SHADOW_TILT_V) + SHADOW_DISTANCE) // vertical extension due to shadow offset
#define H_EXTENDED (H_RES + 2 * H_EXTENSION) // required vertical dimension due to shadow offset
#define V_EXTENDED (V_RES + 2 * V_EXTENSION) // required horizontal dimension due to shadow offset
#define MASK_COORDINATES(x, y) ((y) * H_EXTENDED + (x)) // linearized coordinates of mask
#define SHADOW_COORDINATES(x, y) ((y) * H_EXTENDED + (x)) // linearized coordinates of shadow
#define IMAGE_COORDINATES(x, y) ((y) * H_RES + (x)) // linearized coordinates of images
#define SLICES (2 * ceil((float)SHADOW_DISTANCE / BLOCK_DIM) + 1) // number of slices needed to avoid memory collisions
#define COARSENING_FACTOR 128 // number of outside pixel assignments done by the same thread

// TODO try precalculating SLICES

/// The first iteration generates a black and white image (mask) where
/// pixels inside (divergent) the fractal are black and pixels outside
/// (convergent) are white.
__global__ void __compute_mask(
    const complex c,
    byte *__restrict__ mask);

/// The second iteration plots a circular shadow for each outside pixel of
/// the just generated fractal mask.
/// This is done by adding one to the corresponding elements of the shadow
/// integer array (sized like fractal mask).
/// The higher is the number, the higher is the shadow intensity.
__global__ void __apply_shadow(
    const int h_slice,
    const int v_slice,
    const byte *__restrict__ mask,
    int *__restrict__ shadow);

/// This iteration assigns the inside image.
/// The shadow toner for the inner image is computed starting from the corresponding
/// value in the shadow array.
__global__ void __assign_final_in(
    const int in_len,
    const int *__restrict__ in_pixel,
    const int *__restrict__ shadow,
    const byte *__restrict__ inside,
    byte *image);

/// This iteration assigns the outside image.
__global__ void __assign_final_out(
    const int out_len,
    const int *__restrict__ out_pixel,
    const byte *__restrict__ outside,
    byte *image);

__host__ void generate_art(const complex *c, byte *image, const byte *inside, const byte *outside) {

    // Initialize events
    float time;
    double start_ms, stop_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory
    byte *mask_h, *mask_d, *inside_d, *outside_d, *image_d;
    int *shadow_d, *in_pixel_h, *out_pixel_h, *in_pixel_d, *out_pixel_d;
    cudaMalloc(&mask_d, V_EXTENDED * H_EXTENDED * sizeof(byte));
    cudaMalloc(&shadow_d, V_EXTENDED * H_EXTENDED * sizeof(int));
    cudaMalloc(&inside_d, 3 * V_RES * H_RES * sizeof(byte));
    cudaMalloc(&outside_d, 3 * V_RES * H_RES * sizeof(byte));
    cudaMalloc(&image_d, 3 * V_RES * H_RES * sizeof(byte));
    cudaMalloc(&in_pixel_d, V_EXTENDED * H_EXTENDED * sizeof(int));
    cudaMalloc(&out_pixel_d, V_EXTENDED * H_EXTENDED * sizeof(int));
    mask_h = (byte *)malloc(V_EXTENDED * H_EXTENDED * sizeof(byte));
    in_pixel_h = (int *)malloc(V_EXTENDED * H_EXTENDED * sizeof(int));
    out_pixel_h = (int *)malloc(V_EXTENDED * H_EXTENDED * sizeof(int));

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
    grid_size = dim3(
        ceil((float)H_EXTENDED / block_size.x),
        ceil((float)V_EXTENDED / block_size.y));
    __compute_mask << <grid_size, block_size >> > (*c, mask_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL
        cudaEventElapsedTime(&time, start, stop);
    printf("Mask computation: %f\n", time);

    // For each pixel compute shadow value
    cudaEventRecord(start);
    for (int i = 0; i < SLICES; i++)
        for (int j = 0; j < SLICES; j++) {
            grid_size = dim3( // TODO try using normal approximation instead of ceil
                ceil((float)(ceil((float)H_EXTENDED / block_size.x) - i) / SLICES),
                ceil((float)(ceil((float)V_EXTENDED / block_size.y) - j) / SLICES));
            __apply_shadow << <grid_size, block_size >> > (i, j, mask_d, shadow_d);
            cudaDeviceSynchronize();
            CHECK_KERNELCALL
        }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    printf("Shadow application: %f\n", time);

    // Divide mask elements into in e out array
    cudaMemcpy(mask_h, mask_d, V_EXTENDED * H_EXTENDED * sizeof(byte), cudaMemcpyDeviceToHost);
    int in = 0, out = 0;
    for (int i = 0; i < H_RES; i++)
        for (int j = 0; j < V_RES; j++)
            if (mask_h[MASK_COORDINATES(i + H_EXTENSION, j + V_EXTENSION)] == IN)
                in_pixel_h[in++] = IMAGE_COORDINATES(i, j);
            else
                out_pixel_h[out++] = IMAGE_COORDINATES(i, j);
    cudaMemcpy(in_pixel_d, in_pixel_h, V_EXTENDED * H_EXTENDED * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_pixel_d, out_pixel_h, V_EXTENDED * H_EXTENDED * sizeof(int), cudaMemcpyHostToDevice);

    // For each pixel select final image, computing its shadow
    cudaEventRecord(start);
    grid_size = dim3(
        ceil(sqrt((float)in / block_size.x)),
        ceil(sqrt((float)in / block_size.y)));
    __assign_final_in << <grid_size, block_size >> > (in, in_pixel_d, shadow_d, inside_d, image_d);
    grid_size = dim3(
        ceil(sqrt((float)out / COARSENING_FACTOR / block_size.x)),
        ceil(sqrt((float)out / COARSENING_FACTOR / block_size.y)));
    __assign_final_out << <grid_size, block_size >> > (out, out_pixel_d, outside_d, image_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL
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
}

__global__ void __compute_mask(
    const complex c,
    byte *__restrict__ mask) {

    // Calculate coordinates of the pixel
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= H_EXTENDED || v >= V_EXTENDED) return;

    // Calculate the side length of a pixel in the complex plane
    double res_unit = (double)SCALE / (H_RES / 2);

    // Calculate coordinates of the pixel in the complex plane
    complex z0, z1;
    z0.r = res_unit * (h - H_EXTENDED / 2) + CENTER_X;
    z0.i = res_unit * (v - V_EXTENDED / 2) + CENTER_Y;

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
            mask[MASK_COORDINATES(h, v)] = OUT;
            return;
        }
    }
}

__global__ void __apply_shadow(
    const int h_slice,
    const int v_slice,
    const byte *__restrict__ mask,
    int *__restrict__ shadow) {
#define SHADOW_TILE_DIM (BLOCK_DIM + 2 * SHADOW_DISTANCE) // shared shadow matrix side length

    // Calculate coordinates of the pixel
    int h = (SLICES * blockIdx.x + h_slice) * blockDim.x + threadIdx.x;
    int v = (SLICES * blockIdx.y + v_slice) * blockDim.y + threadIdx.y;

    // Allocate and intialize shared space
    __shared__ unsigned short shadow_tile[SHADOW_TILE_DIM][SHADOW_TILE_DIM];
    for (int i = threadIdx.x; i < SHADOW_TILE_DIM; i += BLOCK_DIM)
        for (int j = threadIdx.y; j < SHADOW_TILE_DIM; j += BLOCK_DIM)
            shadow_tile[i][j] = 0;

    // Check boundaries and ignore points in the image below
    bool plot = h < H_EXTENDED && v < V_EXTENDED && mask[MASK_COORDINATES(h, v)] == OUT;

    // Plot a circular shadow
    for (int i = -SHADOW_DISTANCE; i <= SHADOW_DISTANCE; i++) {
        for (int j = -SHADOW_DISTANCE; j <= SHADOW_DISTANCE; j++) {
            __syncthreads();

            // Increment shadow value if the current shadow index is inside borders and radius
            if (plot && h + i < H_EXTENDED && h + i >= 0 && v + j < V_EXTENDED && v + j >= 0 &&
                i * i + j * j < SHADOW_DISTANCE * SHADOW_DISTANCE)
                shadow_tile[SHADOW_DISTANCE + threadIdx.x + i][SHADOW_DISTANCE + threadIdx.y + j]++;
        }
    }

    __syncthreads();

    // Update global memory with shadow values
    for (int i = threadIdx.x; i < SHADOW_TILE_DIM; i += BLOCK_DIM)
        for (int j = threadIdx.y; j < SHADOW_TILE_DIM; j += BLOCK_DIM) {
            int x = h - threadIdx.x - SHADOW_DISTANCE + i;
            int y = v - threadIdx.y - SHADOW_DISTANCE + j;
            if (x >= 0 && x < H_EXTENDED && y >= 0 && y < V_EXTENDED)
                shadow[SHADOW_COORDINATES(x, y)] += shadow_tile[i][j];
        }

#undef SHADOW_TILE_DIM
}

__global__ void __assign_final_in(
    const int in_len,
    const int *__restrict__ in_pixel,
    const int *__restrict__ shadow,
    const byte *__restrict__ inside,
    byte *image) {

    // Calculate index of the pixel
    int i = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in_len) return;
    int image_idx = in_pixel[i];

    // Shadow intensity computation
    float toner = (exp(-SHADOW_SHARPNESS *
        shadow[SHADOW_COORDINATES(
            image_idx % H_RES + H_EXTENSION + (SHADOW_TILT_H),
            image_idx / H_RES + V_EXTENSION + (SHADOW_TILT_V)
        )] /
        (3.1416 * SHADOW_DISTANCE * SHADOW_DISTANCE))) *
        SHADOW_INTENSITY + (1 - SHADOW_INTENSITY);

    // Inside image assignment with shadow
    image[3 * image_idx + 0] = inside[3 * image_idx + 0] * toner;
    image[3 * image_idx + 1] = inside[3 * image_idx + 1] * toner;
    image[3 * image_idx + 2] = inside[3 * image_idx + 2] * toner;
}

__global__ void __assign_final_out(
    const int out_len,
    const int *__restrict__ out_pixel,
    const byte *__restrict__ outside,
    byte *image) {

    // Calculate index of the pixel
    int i = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    // Compute assignments on consecutive pixels
    for (; i < out_len; i += out_len / COARSENING_FACTOR) {
        int image_idx = out_pixel[i];

        // Outside image assignment
        image[3 * image_idx + 0] = outside[3 * image_idx + 0];
        image[3 * image_idx + 1] = outside[3 * image_idx + 1];
        image[3 * image_idx + 2] = outside[3 * image_idx + 2];

        i++;
    }
}

// nothing

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
