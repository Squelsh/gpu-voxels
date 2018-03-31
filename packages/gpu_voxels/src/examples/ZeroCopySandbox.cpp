// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2018-03-12
 *
 * Proof of concept for the ZeroCopy mechanism on hardware that uses a shared
 * memory architecture between CPU and GPU.
 * The PointCloudTX class is a stripped version of the regualar GPU-Voxels PointCloud.
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/PointCloudTX.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;


int main(int argc, char* argv[])
{

  icl_core::logging::initialize(argc, argv);

  // Set flag to enable zero copy access
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // Arrays
  Vector3f* points_in  = NULL;
  Vector3f* points_out  = NULL;
  Matrix4f* trafo = NULL;

  // Allocate host memory using CUDA allocation calls
  cudaHostAlloc((void **)&points_in,  10000 * sizeof(Vector3f), cudaHostAllocMapped);
  cudaHostAlloc((void **)&points_out, 10000 * sizeof(Vector3f), cudaHostAllocMapped);
  cudaHostAlloc((void **)&trafo, sizeof(Matrix4f), cudaHostAllocMapped);

  for(int i = 0; i < 10000; i++)
  {
    points_in[i] = Vector3f(i*0.04, i*0.03, i*0.02);
  }

  *trafo = Matrix4f::createFromRotationAndTranslation(Matrix3f::createFromRPY(1.57,0,0), Vector3f(0,0,0) );

  PointCloudTX pc;

  pc.transform(trafo, points_in, points_out, 10000);

  for(int i = 0; i < 10000; i++)
  {
    std::cout << i << " : I " << points_in[i] << std::endl;
    std::cout << i << " : O " << points_out[i] << std::endl;
  }

  sleep(1);
  return 0;
}

