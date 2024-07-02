#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>

#define H_RES 3840 // horizontal resolution
#define V_RES 2160 // vertical resolution
#define CENTER_X 0 // X coordinate for image center
#define CENTER_Y 0 // Y coordinate for image center
#define SCALE 2 // maximum X value in the fractal graph
#define ITERATIONS (1 << 8) // number of iteration for checking divergence
#define R 2 // ceiling upon which function is considered divergent
#define SHADOW_DISTANCE 64 // radius of the circular shadow plot
#define SHADOW_SHARPNESS 1 // rapidity with which shadow gets dark
#define SHADOW_TILT_H -64 // horizontal offset from where shadow is plotted
#define SHADOW_TILT_V 32 // vertical offset from where shadow is plotted
#define SHADOW_INTENSITY 0.8 // blackness of the shadow
#define BLOCK_DIM_CM 4 // threads per block dimension (compute mask kernel)
#define BLOCK_DIM_AS 26 // threads per block dimension (apply shadow kernel)
#define BLOCK_DIM_AF 16 // threads per block dimension (assign final kernel)
#define OUT 0xFF // outside color of the fractal mask
#define IN 0x00 // inside color of the fractal mask
#define H_EXTENSION (abs(SHADOW_TILT_H) + SHADOW_DISTANCE) // horizontal extension due to shadow offset
#define V_EXTENSION (abs(SHADOW_TILT_V) + SHADOW_DISTANCE) // vertical extension due to shadow offset
#define H_EXTENDED (H_RES + 2 * H_EXTENSION) // required vertical dimension due to shadow offset
#define V_EXTENDED (V_RES + 2 * V_EXTENSION) // required horizontal dimension due to shadow offset
#define MASK_COORDINATES(x, y) ((y) * H_EXTENDED + (x)) // linearized coordinates of mask
#define SHADOW_COORDINATES(x, y) ((y) * H_EXTENDED + (x)) // linearized coordinates of shadow
#define IMAGE_COORDINATES(x, y) ((y) * H_RES + (x)) // linearized coordinates of images
#define SLICES (2 * (SHADOW_DISTANCE + BLOCK_DIM_AS - 1) / BLOCK_DIM_AS + 1) // number of slices needed to avoid memory collisions
#define COARSE_BLOCK (1 << 8) // pixel block dimension assigned to a single thread in the first iteration
#define COARSE_FACTOR (1 << 4) // division factor on pixel block size at each coarsening iteration
#define COARSE_THRESHOLD (1 << 4) // minimum coarse block dimension before full independent pixel computations

/// Compile time check that coarse block is a power of coarse factor
constexpr int log2(int x) { return x == 1 ? 0 : 1 + log2(x / 2); }
static_assert(log2(COARSE_BLOCK) % log2(COARSE_FACTOR) == 0, "coarsening: block must be power of factor");

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
__host__ void generate_art(const complex *c_param, byte *image, const byte *inside, const byte *outside);

/// The first iteration generates a black and white image (mask) where
/// pixels inside (divergent) the fractal are black and pixels outside
/// (convergent) are white.
__global__ void compute_mask(
    const int hpin,
    const int vpin,
    const int coarse_size);

/// Fractal value is computed on the pixel coordinates provided. Mask may be then updated accordingly.
__device__ byte compute_pixel(const int h, const int v, const bool update);

/// Coarse block border is computed, and if color is equal to all pixels, return true.
__device__ bool common_border(const int hpin, const int vpin, const int coarse_size, byte *fill);

/// When checking if the pixels in the border of a block have same value, that is done in parallel
__global__ void border_pixel(const int hpin, const int vpin, const int coarse_size, byte fill, bool *outcome);

/// If block has same value for each pixel in the border, color is filled all together
__global__ void fill_block(const int hpin, const int vpin, const byte fill);

/// If coarse block stands under minimum size, each pixel is computed independently
__global__ void compute_block(const int hpin, const int vpin);

/// The second iteration plots a circular shadow for each outside pixel of
/// the just generated fractal mask.
/// This is done by adding one to the corresponding elements of the shadow
/// integer array (sized like fractal mask).
/// The higher is the number, the higher is the shadow intensity.
__global__ void apply_shadow(
    const int h_slice,
    const int v_slice,
    const byte *__restrict__ mask,
    int *__restrict__ shadow);

/// The third iteration assigns the inside and the outside images.
/// The shadow toner for the inner image is computed starting from the corresponding
/// value in the shadow array.
__global__ void assign_final(
    const int *__restrict__ shadow,
    const byte *__restrict__ mask,
    const byte *__restrict__ inside,
    const byte *__restrict__ outside,
    byte *__restrict__ image);

/// Constant c value of fractal function
__constant__ complex c;

/// Mask address global value used to avoid useless parameter passing in recursive kernel
__device__ byte *mask;

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
    if (argc <= 2) {
        printf("Provide a complex number\n");
        return 1;
    }
    complex c;
    c.r = strtod(argv[1], NULL);
    c.i = strtod(argv[2], NULL);

    // Allocate memory for images
    byte *image = (byte *)malloc(3 * V_RES * H_RES * sizeof(byte));
    byte *inside = (byte *)malloc(3 * V_RES * H_RES * sizeof(byte));
    byte *outside = (byte *)malloc(3 * V_RES * H_RES * sizeof(byte));

    if (argc >= 5) {

        // Retrive color values
        byte colors[6];
        char temp[3];
        temp[2] = '\0';
        for (int i = 0; i < 6; i++) {
            temp[0] = argv[3 + i / 3][(i % 3) * 2];
            temp[1] = argv[3 + i / 3][(i % 3) * 2 + 1];
            colors[i] = (byte)strtoul(temp, NULL, 16);
        }

        // Load input colors as input images
        for (int i = 0; i < V_RES * H_RES; i++)
            for (int j = 0; j < 3; j++) {
                inside[3 * i + j] = colors[j];
                outside[3 * i + j] = colors[3 + j];
            }


    } else {

        // Load the two input images
        if (load_image("inside.ppm", inside) < 0) {
            fprintf(stderr, "Error opening %s\n", "inside.ppm");
            return 1;
        }
        if (load_image("outside.ppm", outside) < 0) {
            fprintf(stderr, "Error opening %s\n", "outside.ppm");
            return 1;
        }

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

__host__ void generate_art(const complex *c_param, byte *image, const byte *inside, const byte *outside) {

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
    cudaMemcpyToSymbol(c, c_param, sizeof(complex));
    cudaMemcpyToSymbol(mask, &mask_d, sizeof(byte *));
    cudaMemcpy(inside_d, inside, 3 * V_RES * H_RES * sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(outside_d, outside, 3 * V_RES * H_RES * sizeof(byte), cudaMemcpyHostToDevice);
    stop_ms = milliseconds();
    printf("Memory transfer to device: %f\n", stop_ms - start_ms);

    dim3 block_size, grid_size;

    // For each pixel compute fractal mask
    cudaEventRecord(start);
    block_size = dim3(BLOCK_DIM_CM, BLOCK_DIM_CM);
    grid_size = dim3(
        ceil((float)H_EXTENDED / block_size.x / COARSE_BLOCK),
        ceil((float)V_EXTENDED / block_size.y / COARSE_BLOCK));
    compute_mask << <grid_size, block_size >> > (0, 0, COARSE_BLOCK);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL;
    cudaEventElapsedTime(&time, start, stop);
    printf("Mask computation: %f\n", time);

    // For each pixel compute shadow value
    cudaEventRecord(start);
    block_size = dim3(BLOCK_DIM_AS, BLOCK_DIM_AS);
    for (int i = 0; i < SLICES; i++)
        for (int j = 0; j < SLICES; j++) {
            grid_size = dim3(
                ceil((float)(round((float)H_EXTENDED / block_size.x) - i) / SLICES),
                ceil((float)(round((float)V_EXTENDED / block_size.y) - j) / SLICES));
            apply_shadow << <grid_size, block_size >> > (i, j, mask_d, shadow_d);
            cudaDeviceSynchronize();
            CHECK_KERNELCALL;
        }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time, start, stop);
    printf("Shadow application: %f\n", time);

    // For each pixel select final image, computing its shadow
    cudaEventRecord(start);
    block_size = dim3(BLOCK_DIM_AF, BLOCK_DIM_AF);
    grid_size = dim3(
        ceil((float)H_RES / block_size.x),
        ceil((float)V_RES / block_size.y));
    assign_final << <grid_size, block_size >> > (shadow_d, mask_d, inside_d, outside_d, image_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    CHECK_KERNELCALL;
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

__global__ void compute_mask(
    const int hpin,
    const int vpin,
    const int coarse_size) {

    // Calculate coordinates of the pixel
    int h = (blockIdx.x * blockDim.x + threadIdx.x) * coarse_size + hpin;
    int v = (blockIdx.y * blockDim.y + threadIdx.y) * coarse_size + vpin;

    byte fill;

    if (common_border(h, v, coarse_size, &fill)) {

        // Coarse block has the same outcome for each pixel inside
        dim3 block_size(8, 8);
        dim3 grid_size(
            coarse_size / block_size.x,
            coarse_size / block_size.y);
        fill_block << <grid_size, block_size >> > (h, v, fill);

    } else if (coarse_size <= COARSE_THRESHOLD) {

        // Coarse block is minimal size, i.e. pixel are processed independently
        dim3 block_size(8, 8);
        dim3 grid_size(
            coarse_size / block_size.x,
            coarse_size / block_size.y);
        compute_block << <grid_size, block_size >> > (h, v);

    } else {

        // Coarse block has heterogenous computation efforts
        dim3 block_size(4, 4);
        dim3 grid_size(
            COARSE_FACTOR / block_size.x,
            COARSE_FACTOR / block_size.y);
        compute_mask << <grid_size, block_size >> > (h, v, coarse_size / COARSE_FACTOR);

    }
}

__device__ byte compute_pixel(const int h, const int v, const bool update) {
#define RES_UNIT ((double)SCALE / (H_RES / 2)) // side length of a pixel in the complex plane

    // Check boundaries
    if (h >= H_EXTENDED || v >= V_EXTENDED) return OUT;

    // Calculate coordinates of the pixel in the complex plane
    complex z0, z1;
    z0.r = RES_UNIT * (h - H_EXTENDED / 2) + CENTER_X;
    z0.i = RES_UNIT * (v - V_EXTENDED / 2) + CENTER_Y;

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
            if (update) mask[MASK_COORDINATES(h, v)] = OUT;
            return OUT;
        }
    }

    return IN;

#undef RES_UNIT
}

__device__ bool common_border(const int hpin, const int vpin, const int coarse_size, byte *fill) {

    // Calculate color for first pixel
    byte temp = compute_pixel(hpin, vpin, false);

    // Check if other vertices have the same color
    if (
        temp != compute_pixel(hpin + coarse_size - 1, vpin + coarse_size - 1, false) ||
        temp != compute_pixel(hpin + coarse_size - 1, vpin, false) ||
        temp != compute_pixel(hpin, vpin + coarse_size - 1, false)
        ) return false;

    // Check block side pixels
    bool *outcome = (bool *)malloc(sizeof(bool));
    *outcome = true;
    border_pixel << <4, coarse_size - 1 >> > (hpin, vpin, coarse_size, temp, outcome);
    cudaDeviceSynchronize();

    // If all border's pixels require same color, return true
    bool res = *outcome;
    free(outcome);
    *fill = temp;
    return res;
}

__global__ void border_pixel(const int hpin, const int vpin, const int coarse_size, byte fill, bool *outcome) {

    // Calculate coordinates of the pixel
    int h = hpin + (coarse_size - 1) * (blockIdx.x == 0) + threadIdx.x * ((blockIdx.x % 2) == 1);
    int v = vpin + (coarse_size - 1) * (blockIdx.x == 1) + threadIdx.x * ((blockIdx.x % 2) == 0);

    // Check pixel outcome
    if (fill != compute_pixel(h, v, false)) *outcome = false;
}

__global__ void fill_block(const int hpin, const int vpin, const byte fill) {

    // Calculate coordinates of the pixel
    int h = blockIdx.x * blockDim.x + threadIdx.x + hpin;
    int v = blockIdx.y * blockDim.y + threadIdx.y + vpin;

    // Check boundaries and fill color
    if (h < H_EXTENDED && v < V_EXTENDED)
        mask[MASK_COORDINATES(h, v)] = fill;
}

__global__ void compute_block(const int hpin, const int vpin) {

    // Calculate coordinates of the pixel
    int h = blockIdx.x * blockDim.x + threadIdx.x + hpin;
    int v = blockIdx.y * blockDim.y + threadIdx.y + vpin;

    // Check boundaries and compute pixel
    if (h < H_EXTENDED && v < V_EXTENDED)
        compute_pixel(h, v, true);
}

__global__ void apply_shadow(
    const int h_slice,
    const int v_slice,
    const byte *__restrict__ mask,
    int *__restrict__ shadow) {
#define SHADOW_TILE_DIM (BLOCK_DIM_AS + 2 * SHADOW_DISTANCE) // shared shadow matrix side length

    // Calculate coordinates of the pixel
    int h = (SLICES * blockIdx.x + h_slice) * blockDim.x + threadIdx.x;
    int v = (SLICES * blockIdx.y + v_slice) * blockDim.y + threadIdx.y;

    // Allocate and intialize shared space
    __shared__ unsigned short shadow_tile[SHADOW_TILE_DIM][SHADOW_TILE_DIM];
    for (int i = threadIdx.x; i < SHADOW_TILE_DIM; i += BLOCK_DIM_AS)
        for (int j = threadIdx.y; j < SHADOW_TILE_DIM; j += BLOCK_DIM_AS)
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
    for (int i = threadIdx.x; i < SHADOW_TILE_DIM; i += BLOCK_DIM_AS)
        for (int j = threadIdx.y; j < SHADOW_TILE_DIM; j += BLOCK_DIM_AS) {
            int x = h - threadIdx.x - SHADOW_DISTANCE + i;
            int y = v - threadIdx.y - SHADOW_DISTANCE + j;
            if (x >= 0 && x < H_EXTENDED && y >= 0 && y < V_EXTENDED)
                shadow[SHADOW_COORDINATES(x, y)] += shadow_tile[i][j];
        }

#undef SHADOW_TILE_DIM
}

__global__ void assign_final(
    const int *__restrict__ shadow,
    const byte *__restrict__ mask,
    const byte *__restrict__ inside,
    const byte *__restrict__ outside,
    byte *__restrict__ image) {

    // Calculate coordinates of the pixel
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (h >= H_RES || v >= V_RES) return;

    // Calculate index of the pixel
    int image_idx = IMAGE_COORDINATES(h, v);

    if (mask[MASK_COORDINATES(H_EXTENSION + h, V_EXTENSION + v)] == OUT) {

        // Outside image assignment
        image[3 * image_idx + 0] = outside[3 * image_idx + 0];
        image[3 * image_idx + 1] = outside[3 * image_idx + 1];
        image[3 * image_idx + 2] = outside[3 * image_idx + 2];

    } else {

        // Shadow intensity computation
        float toner = (exp(-SHADOW_SHARPNESS *
            shadow[SHADOW_COORDINATES(
                H_EXTENSION + h + (SHADOW_TILT_H),
                V_EXTENSION + v + (SHADOW_TILT_V))] /
            (3.1416 * SHADOW_DISTANCE * SHADOW_DISTANCE))) *
            SHADOW_INTENSITY + (1 - SHADOW_INTENSITY);

        // Inside image assignment with shadow
        image[3 * image_idx + 0] = inside[3 * image_idx + 0] * toner;
        image[3 * image_idx + 1] = inside[3 * image_idx + 1] * toner;
        image[3 * image_idx + 2] = inside[3 * image_idx + 2] * toner;

    }
}

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
