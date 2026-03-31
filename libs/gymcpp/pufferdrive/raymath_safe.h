//
// Created by jaeger on 10/25/25.
// There is a nasty conflict between ray and cuda which both try to define float3.
// This file is a workaround.

#ifndef PPO_CPP_RAYMATH_SAFE_H
#define PPO_CPP_RAYMATH_SAFE_H

#define float3 ray_float3
#include "raymath.h"
#undef float3

#endif //PPO_CPP_RAYMATH_SAFE_H