-------------- Setup: --------------

1. From the root directiory (the same directory as this README.txt file), obtain the diffusion module by running the following:
git clone https://github.com/luost26/diffusion-point-cloud
mv diffusion-point-cloud ModelDiffusion/diffusion

2. Navigate to https://drive.google.com/drive/folders/1Su0hCuGFo1AGrNb_VMNnlF7qeQwKjfhZ
Download pretrained/GEN_airplane.pt into ./ModelDiffusion/diffusion/pretrained
Download data/shapenet.hdf5 into ./ModelDiffusion/diffusion/data

3. Generate a point cloud dataset (only used for generating the dataset of images of point clouds) by running the following:
cd ModelDiffusion/diffusion
python3 test_gen.py --ckpt ./pretrained/GEN_airplane.pt --categories airplane
cd ../..


-------------- Testing: --------------
If you want to visualize the model throughout training, run mkdir Comparisons and change "visualize_validation=False" in ImgMatching.ipynb to "visualize_validation=True", otherwise you can safely ignore this step

Then just open ./ImgMatching.ipynb and run