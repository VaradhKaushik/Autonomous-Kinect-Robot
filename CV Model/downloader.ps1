for ($i = 5; $i -le 14; $i++) {
    wget -OutFile "C:\Users\pokes\OneDrive\Documents\CS4610\CS5335-4610-Project\tars\food_can_$i.tar" -Uri "https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/food_can_$i.tar"
}

for ($i = 4; $i -le 12; $i++) {
    wget -OutFile "C:\Users\pokes\OneDrive\Documents\CS4610\CS5335-4610-Project\tars\food_box_$i.tar" -Uri "https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_full/food_box_$i.tar"
}