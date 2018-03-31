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
 * Class holding and manipulating PointClouds
 * as Arrays of Vector3fs via CUDA memory shared between host and device.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_PointCloudTX_H_INCLUDED
#define GPU_VOXELS_HELPERS_PointCloudTX_H_INCLUDED

#include <stdint.h> // for fixed size datatypes
#include <vector>
#include <cuda_runtime.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/MathHelpers.h>

namespace gpu_voxels
{

class PointCloudTX
{
public:

  /*!
   * \brief PointCloudTX Constructs empty PointCloudTX
   */
  PointCloudTX();
 
  ~PointCloudTX();

  /*!
   * \brief transform Applies a transformation to this cloud and returns the result in transformed_cloud
   * \param transform Transformation matrix
   * \param transformed_cloud Output cloud. Will be resized.
   */
  void transform(const Matrix4f* transform, Vector3f* input_cloud, Vector3f* transformed_cloud, size_t num_points);


private:

  mutable Matrix4f* m_transformation_dev;
  mutable uint32_t m_blocks;
  mutable uint32_t m_threads_per_block;
};

}//end namespace gpu_voxels
#endif // GPU_VOXELS_HELPERS_PointCloudTX_H_INCLUDED
