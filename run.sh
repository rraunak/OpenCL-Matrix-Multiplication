#!/bin/bash

./matrix_mul 512 16 CPU NORMAL
./matrix_mul 512 8 CPU TILED
./matrix_mul 512 16 CPU TILED

./matrix_mul 1024 16 CPU NORMAL
./matrix_mul 1024 8 CPU TILED
./matrix_mul 1024 16 CPU TILED

./matrix_mul 2048 16 CPU NORMAL
./matrix_mul 2048 8 CPU TILED
./matrix_mul 2048 16 CPU TILED

./matrix_mul 512 16 GPU NORMAL
./matrix_mul 512 8 GPU TILED
./matrix_mul 512 16 GPU TILED

./matrix_mul 1024 16 GPU NORMAL
./matrix_mul 1024 8 GPU TILED
./matrix_mul 1024 16 GPU TILED

./matrix_mul 2048 16 GPU NORMAL
./matrix_mul 2048 8 GPU TILED
./matrix_mul 2048 16 GPU TILED
