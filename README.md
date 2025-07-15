# ppo.cpp
This repository implements PPO for continuous action spaces in C++ matching the [CleanRL](https://github.com/vwxyzjn/cleanrl) python implementation closely. It also contains a [minimum port](libs/gymcpp) of [gymnasium](https://github.com/Farama-Foundation/Gymnasium) to C++, containing the needed functionality for PPO.
Additionally, the repository provides environments for mujoco, [half_cheetah_v5](libs/gymcpp/mujoco/half_cheetah_v5.h) and [humanoid_v4](libs/gymcpp/mujoco/humanoid_v4.h), as well as an environment for autonomous driving with the CARLA leaderboard 2.0.

The repository also implements Asynchronous Collection Proximal Policy Optimization (AC-PPO) which parallelizes data collection via multithreading and cuda streams, leading to faster training time than PPO in nonhomogeneous environments.
The idea is described in Appendix B.1 of this [paper](https://arxiv.org/abs/2504.17838).

To run the training and evaluation of with the CARLA leaderboard 2.0, you also need to download and set up the [CaRL repo](https://github.com/autonomousvision/CaRL).

## Setup
To most convenient way to compile and run the program is to build the singularity container and run the code inside that.
Building the container can take a while, depending on your CPU power, because it builds several libraries and needs 12 GB of space. I have tested the code with singularity-ce version 3.11, but other version should work as well.
```Shell
cd tools
sudo singularity build ppo_cpp.sif make_singularity_image.def
```
Alternatively you can setup your own computer by installing all necessary libaries. You can have a look at [make_singularity_image.def](tools/make_singularity_image.def) on how to do it.
This often takes some time and you will face various issues, so it is only recommended for experienced C++ users.

The container is currently not compatible for the Mujoco code. Some libaries are missing and the use of 
`<format>` in the file needs to be changed to `<boost/format>` because the ubuntu 22 gcc version in the compiler does not support `<format>` yet.


## Compiling
The code can be compiled via cmake:
```Shell
cd /path/to/ppo.cpp
singularity exec /path/to/ppo_cpp.sif cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Ninja"
singularity exec /path/to/ppo_cpp.sif cmake --build build -j20
```
By default, the cmake script compiles the CARLA code. 
The mujoco code is currently commented in [CMakeLists.txt](CMakeLists.txt) because the container doesn't support it yet.

## Training models
To train CARLA models have a look at the training scripts in [CaRL](https://github.com/autonomousvision/CaRL/blob/main/CARLA/team_code/train_carl_cpp.sh).  
Generally you need to build the container, compile the program and then set the paths correctly: 
```Shell
--ppo_cpp_install_path /path/to/folder_with_binaries
--cpp_singularity_file_path /path/to/ppo_cpp.sif
```

Libtorch does not natively support multi-gpu training. 
We implemented the multi-gpu communication ourselves using the backend code of [torch-fort](https://github.com/NVIDIA/TorchFort).
To use multiple GPUs for training the code needs to be started with mpirun (`-n` = number of GPUs), similar how pytorch DDP ist started with torchrun:
```Shell
mpirun -n 1 --bind-to none /path/to/ppo_cpp --exp_name_stem Test_AC_PPO_000 --env_id HalfCheetah-v5 --seed 500
```

## Reproducibility
We implemented the Mujoco environments mainly to check if the implementation is correct.
Below we compare the runs of [ppo_continuous_action.cpp](src/ppo_continuous_action.cpp) with cleanRL's [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py).
There are some numerical differences but the runs are very close for RL standards.

![HalfCheetah](./docs/halfcheetah_clearnrl_ppo_cpp.png)

![Humanoid](./docs/humanoid_clearnrl_ppo_cpp.png)

Interestingly the C++ implementation is up to 72% faster in SPS than the python implementation on the same hardware without any specific optimizations.
All runs are in CPU mode which is faster than GPU mode for these tiny Mujoco networks in both languages, the GPU default in ppo_continuous_action.py is suboptimal.
For the larger CNN used in the CaRL model the speedup of the GPU in enough to outweigh the GPU-CPU communication overhead.

## Known problems

The code implements the DD-PPO preemption trick.
I had some runs with poor performance using the preemption trick which is why I disabled it. 
It could be that the trick itself caused the performance degradation but there might also be a bug, so I do not recommend using it right now (use_dd_ppo_preempt=0).

The CARLA training code seems to have higher peak GPU memory usage than our pytorch version.
My training runs did not max out my GPU memory, so I have not investigated this issue further.
There might be some PyTorch DDP memory optimization that is not included in our custom implementation or something like that.

## License

The original code in this repository is provided under the Civil-M license, which is a variant of the MIT license that bans dual-use. [The license](LICENSE) contains a partial copyleft which requires derivative work to include the civil clause in their license. For further information see the accompaning documentation on [Civil Software Licenses](docs/Jaeger2025LicenseWhitepaper.pdf).

## Citation
If you find the repo useful, please consider giving it a star &#127775;.
To cite the paper please use the following bibtex:
```BibTeX
@article{Jaeger2025ArXiv, 
        author = {Bernhard Jaeger and Daniel Dauner and Jens Beißwenger and Simon Gerstenecker and Kashyap Chitta and Andreas Geiger}, 
        title = {CaRL: Learning Scalable Planning Policies with Simple Rewards}, 
        year = {2025}, 
        journal = {arXiv.org}, 
        volume = {2504.17838}, 
}
```

## Acknowledgements
The original code in this repository was written by Bernhard Jaeger.

Code like this is build on the shoulders of many other open source repositories.
Particularly, we would like to thank the following repositories for their contributions:

* [clean_rl](https://github.com/vwxyzjn/cleanrl/tree/master)
* [torchfort](https://github.com/NVIDIA/TorchFort)
* [envpool](https://github.com/sail-sg/envpool)

We also thank the creators of the numerous libraries we use. Complex projects like this would not be feasible without your contribution.