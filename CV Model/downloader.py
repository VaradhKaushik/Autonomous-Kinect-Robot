import os

for i in range(5,15):
    os.system(f'wget -P ./tars https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/food_can_{i}.tar')

for i in range(4, 13):
    os.system(f'wget -P ./tars https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/food_box_{i}.tar')