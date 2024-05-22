# README

## Project Description

This project consists of a CUDA plugin for Unity, developed using CUDA version 11.4. The aim of the plugin is to provide advanced computation and processing capabilities by leveraging the power of NVIDIA GPUs through CUDA in a Unity environment.

## Requirements

- Unity 2020.3.48f1 or higher
- CUDA 11.4 (or the version you are using, following the compilation instructions)
- Visual Studio 2019 or higher (to recompile the plugin)

## Project Structure

The project mainly consists of the following folders:

- `Assets/`: Contains all the resources and scripts needed to integrate the plugin into Unity.
- `CppCudaPlugin/`: C++ CUDA plugin source code.
- `CudaPlugin/`: CUDA plugin source code.

## Usage Instructions

 - Ensure that NVIDIA GPU drivers and CUDA 11.4 are installed.
 - Open the Unity project using Unity Hub and run it.


### Modifying and Recompiling the Plugin

If you need to modify the plugin's source code (folders `cppcudaplugin` or `cudaplugin`), you must recompile the plugin project to generate a new `.dll` file.

#### Steps to Recompile the Project:

1. Open the plugin project in Visual Studio.
2. Navigate to the project properties.
3. In the project properties > additional include directories, ensure that the paths of your CUDA installation are correctly set.
4. In the project properties > additional include directories, ensure that the path of yor Unity Editor Plugin API is correctly set. 
5. Compile the project to generate the new `.dll` file.
6. There is no need to replace the generated `.dll` file in the Unity plugins folder (`Assets/Plugins`). The '.dll' file is saved in `Assets/Plugins` (see Project Properties > Events > Post-build events). Just make sure the Unity Project is not opened.

