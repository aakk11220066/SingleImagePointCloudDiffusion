import torch


# ----- Utils -----
def thicken_points(mat):
    return mat + torch.roll(mat, 1, 0) + torch.roll(mat, -1, 0) + torch.roll(mat, 1, 1) + torch.roll(mat, -1, 1) \
        + torch.roll(mat, [1,1], [0,1]) + torch.roll(mat, [1,-1], [0,1]) + torch.roll(mat, [-1,1], [0,1]) + torch.roll(mat, [-1,-1], [0,1])

def rotate_x_matrix(angle):
    angle = torch.tensor(angle)
    rotation_matrix = torch.tensor([
        [1,0,0],
        [0,torch.cos(angle),-torch.sin(angle)],
        [0,torch.sin(angle),torch.cos(angle)],
    ])
    return rotation_matrix.T

def rotate_y_matrix(angle):
    angle = torch.tensor(angle)
    rotation_matrix = torch.tensor([
        [torch.cos(angle),0,torch.sin(angle)],
        [0,1,0],
        [-torch.sin(angle),0,torch.cos(angle)],
    ])
    return rotation_matrix.T

def rotate_z_matrix(angle):
    angle = torch.tensor(angle)
    rotation_matrix = torch.tensor([
        [torch.cos(angle),-torch.sin(angle),0],
        [torch.sin(angle),torch.cos(angle),0],
        [0,0,1],
    ])
    return rotation_matrix.T

def rotation_matrix(angle_x, angle_y, angle_z):
    z_rotation = rotate_z_matrix(angle_z)
    y_rotation = rotate_y_matrix(angle_y)
    x_rotation = rotate_x_matrix(angle_x)
    return z_rotation @ y_rotation @ x_rotation

class RotateModule(torch.nn.Module):
    def __init__(self, angle_x, angle_y, angle_z):
        super().__init__()
        
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.register_buffer("rotation_matrix", rotation_matrix(angle_x, angle_y, angle_z).unsqueeze(0))
        
    def __str__(self):
        return f"RotateModule(angle_x={self.angle_x/torch.pi}π, angle_y={self.angle_y/torch.pi}π, angle_z={self.angle_z/torch.pi}π)"
        
    def forward(self, point_cloud):
        return point_cloud @ self.rotation_matrix

# ----- Rasterizer -----
class DifferentiableRaster(torch.nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.width = width
        self.height = height
    
    def forward(self, point_clouds):
        point_clouds_distances = (point_clouds.mT[:,2,:] - point_clouds.mT[:,2,:].min(dim=1).values.unsqueeze(1)) / (point_clouds.mT[:,2,:].max(dim=1).values.unsqueeze(1) - point_clouds.mT[:,2,:].min(dim=1).values.unsqueeze(1))
        point_clouds_distances = torch.maximum(point_clouds_distances, torch.quantile(point_clouds_distances,0.01, dim=1).unsqueeze(1))
        point_clouds_strengths = 1 - point_clouds_distances
        point_clouds_indices = point_clouds.mT[:,:2,:]
        point_clouds_indices = point_clouds_indices - point_clouds_indices.min(dim=2).values.unsqueeze(2)
        point_clouds_indices = point_clouds_indices / point_clouds_indices.max(dim=2).values.unsqueeze(2)
        point_clouds_indices = point_clouds_indices * torch.tensor([[self.height-2],[self.width-2]], device=point_clouds_indices.device).unsqueeze(0) + 1
        # do not make use of borders, for numerical inaccuracy buffer
        point_clouds_index_floors = point_clouds_indices.floor().to(torch.int32)
        point_clouds_index_ceilings = point_clouds_indices.ceil().to(torch.int32)
        
        rasters = []
        for pointcloud_idx in range(point_clouds.shape[0]):
            point_cloud_index_floors = point_clouds_index_floors[pointcloud_idx]
            point_cloud_index_ceilings = point_clouds_index_ceilings[pointcloud_idx]
            point_cloud_indices = point_clouds_indices[pointcloud_idx]
            point_cloud_strengths = point_clouds_strengths[pointcloud_idx]
            
            top_left_raster = torch.sparse_coo_tensor(
                indices=point_cloud_index_floors, 
                values=point_cloud_strengths * torch.prod(point_cloud_indices - point_cloud_index_floors, dim=0),
                size=(self.height, self.width)
            )

            bottom_left_location_coefficient = torch.prod(
                torch.abs(
                    point_cloud_indices - torch.stack([
                        point_cloud_index_ceilings[0,:], 
                        point_cloud_index_floors[1,:]
                    ])
                ),
                dim=0
            )
            bottom_left_raster = torch.sparse_coo_tensor(
                indices=torch.stack([
                    point_cloud_index_ceilings[0,:], 
                    point_cloud_index_floors[1,:]
                ]), 
                values=point_cloud_strengths * bottom_left_location_coefficient,
                size=(self.height, self.width)
            )

            top_right_location_coefficient = torch.prod(
                torch.abs(
                    point_cloud_indices - torch.stack([
                        point_cloud_index_floors[0,:], 
                        point_cloud_index_ceilings[1,:]
                    ])
                ),
                dim=0
            )
            top_right_raster = torch.sparse_coo_tensor(
                indices=torch.stack([
                    point_cloud_index_floors[0,:], 
                    point_cloud_index_ceilings[1,:]
                ]), 
                values=point_cloud_strengths * top_right_location_coefficient,
                size=(self.height, self.width)
            )

            bottom_right_raster = torch.sparse_coo_tensor(
                indices=point_cloud_index_ceilings, 
                values=point_cloud_strengths * torch.prod(point_cloud_index_ceilings - point_cloud_indices, dim=0),
                size=(self.height, self.width)
            )

            rasters.append((top_left_raster + bottom_left_raster + top_right_raster + bottom_right_raster))
        return torch.stack(rasters).to_dense()