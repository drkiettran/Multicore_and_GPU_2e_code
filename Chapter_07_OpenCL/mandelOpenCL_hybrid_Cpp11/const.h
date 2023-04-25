// constants shared by both the host and device code
#pragma once

static const int MAXITER = 1024;
static const int THR_BLK_X = 1; // pixels per thread, x-axis
static const int THR_BLK_Y = 1; // pixels per thread, y-axis
static const int BLOCK_SIDE = 16;       // size of 2D block of threads
