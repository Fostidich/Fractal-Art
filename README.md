# Fractal-Art

The program produces an image starting from the graph of 
a [fractal](https://en.wikipedia.org/wiki/Julia_set#Quadratic_polynomials).

Two input images are incorporated into the final image: the first one is placed inside
the fractal borders, and the other outside.

A shadow effect creates an illusion of depth.

The optimization steps are made in the following order:
- naive;
- tiling;
- slicing;
- coarsening.

###### C/CUDA code written by Francesco Ostidich and Francesco Gangi
