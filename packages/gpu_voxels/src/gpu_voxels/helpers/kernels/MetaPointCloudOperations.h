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
 * \date    2014-06-17
 *
 * MetaPointCloud kernel calls
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_KERNELS_METAPOINTCLOUDOPERATIONS_H_INCLUDED
#define GPU_VOXELS_HELPERS_KERNELS_METAPOINTCLOUDOPERATIONS_H_INCLUDED
#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>

namespace gpu_voxels {

__global__
void kernelDebugMetaPointCloud(MetaPointCloudStruct* meta_point_clouds_struct);




/*!
 * \brief kernelTransformCloud transforms the whole cloud
 * \param transformation_ The transformation to be applied
 * \param input_cloud Original cloud
 * \param transformed_cloud Can be the same as the input_cloud
 */
__global__
void kernelTransformCloud(const Matrix4f* transformation, const MetaPointCloudStruct *input_cloud, MetaPointCloudStruct *transformed_cloud);


/*!
 * \brief kernelTransformCloud transforms a sub cloud of the metaPointCloud
 * \param subcloud_to_transform ID of the subcloud to transform
 * \param transformation_ The transformation to be applied
 * \param input_cloud Original cloud
 * \param transformed_cloud Can be the same as the input_cloud
 */
__global__
void kernelTransformSubCloud(uint8_t subcloud_to_transform, const Matrix4f* transformation_,
                          const MetaPointCloudStruct *input_cloud, MetaPointCloudStruct *transformed_cloud);

}
#endif
