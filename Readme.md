This is the repository for the manuscript: *MLGCN*

# Requirements



You need to configure the following environment before running the MLGCN model. 

It should be noted that this project is carried out in the *Windows system*, if you are using Linux system, We hope you can install the corresponding environment version yourself.

- Windows system
- NVIDIA GeForce RTX 3060Ti
- PyCharm 2022
- python = 3.9.16
- dgl = 1.0.0
- numpy = 1.24.1
- pandas = 2.0.3
- scikit-learn = 1.2.2
- torch = 2.0.1+cu117
- torch-cluster = 1.6.1+pt20cu117
- torch-geometric = 1.7.2
- torch-scatter = 2.1.1+pt20cu117
- torch-sparse = 0.6.17+pt20cu117
- torch-spline-conv = 1.2.2+pt20cu117
- torchaudio = 2.0.2+cu117
- torchdata = 0.7.1
- torchvision = 0.15.2+cu117
- networkx = 3.1
- scipy = 1.10.1

# Usage



After you are ready to run the environment, you can run it in the following way:

1.Download the code from GitHub and unzip it to your own code workspace：

![image-20241217111053079](https://github.com/user-attachments/assets/fae7eb4d-e469-4531-b4da-796c4ec18988)


2.Ensure that you have configured the corresponding operating environment and switched to it：

![image-20241217103053908](https://github.com/user-attachments/assets/543e5b74-6b95-4bdf-853b-cdb9215bad9c)


3.Right click on the code area and click the run button：

![image-20241217111507809](https://github.com/user-attachments/assets/e71280de-a56d-46e7-a877-614a038452c6)


4.Then wait for the running result to appear in the Python console：

![image-20241217111740425](https://github.com/user-attachments/assets/b04b1219-c73d-4a81-9794-6aac7fe348a7)



# File description



MLGCN-GGNet MLGCN-PathNet and MLGCN-PPNet are files for conducting experiments on pan-cancer dataset and three different biological networks.



MLGCN-specific cancer-BLCA MLGCN-specific cancer-BRCA MLGCN-specific cancer-LIHC and MLGCN-specific cancer-LUAD are files for conducting experiments on cancer-specific dataset.
