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
 */
//----------------------------------------------------------------------
#include "PointCloudTX.h"
#include <gpu_voxels/helpers/kernels/MetaPointCloudOperations.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/kernels/HelperOperations.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>

namespace gpu_voxels
{


PointCloudTX::PointCloudTX()
{

}



PointCloudTX::~PointCloudTX()
{

}





void PointCloudTX::transform(const Matrix4f *transform, Vector3f *input_cloud, Vector3f *transformed_cloud, size_t num_points)
{


  Vector3f *transformed_cloud_dev, *input_cloud_dev;
  Matrix4f *m_transformation_dev;
  // Get device pointer from host memory. No allocation or memcpy
  cudaHostGetDevicePointer((void **)&input_cloud_dev, (void *) input_cloud , 0);
  cudaHostGetDevicePointer((void **)&transformed_cloud_dev, (void *) transformed_cloud, 0);
  cudaHostGetDevicePointer((void **)&m_transformation_dev, (void *) transform, 0);


  computeLinearLoad(num_points, &m_blocks, &m_threads_per_block);
  // transform the cloud via Kernel.
  kernelTransformCloud<<< m_blocks, m_threads_per_block >>>
                          (m_transformation_dev,
                           input_cloud_dev,
                           transformed_cloud_dev,
                           num_points);
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}


}// end namespace gpu_voxels
