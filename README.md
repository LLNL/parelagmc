/*
  Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-747639. All rights reserved.
  See the file COPYRIGHT for details. Please also read the file LICENSE.      
                                                                             
  This file is part of the ParELAGMC library. For more information and source 
  code availability see https://github.com/LLNL/parelagmc.                    
                                                                             
  ParELAGMC is free software; you can redistribute it and/or modify it        
  under the terms of the GNU General Public License (as published by the      
  Free Software Foundation) version 2, dated June 1991.  
*/ 
###         Parallel Element Agglomeration Multilevel Monte Carlo Library

                                 version 1.0
```
        ________              ____________________________________  ___________
        ___  __ \_____ __________  ____/__  /___    |_  ____/__   |/  /_  ____/
        __  /_/ /  __ `/_  ___/_  __/  __  / __  /| |  / __ __  /|_/ /_  /
        _  ____// /_/ /_  /   _  /___  _  /___  ___ / /_/ / _  /  / / / /___
        /_/     \__,_/ /_/    /_____/  /_____/_/  |_\____/  /_/  /_/  \____/

```

## Introduction

`ParELAGMC` is a parallel distributed memory C++ library for multilevel 
Monte Carlo (MLMC) simulations with algebraically constructed coarse spaces, 
primarily focusing on generating Gaussian random fields using a novel 
SPDE sampling technique.

`ParELAGMC` enables multilevel variance reduction techniques in the context of
general unstructured meshes by using the specialized element-based
agglomeration techniques implemented in [ParELAG](https://github.com/LLNL/parelag).

The nested hierarchies of algebraically coarse spaces produced by `ParELAG` are then
used to discretize different realizations of the stochastic problem
at different spatial resolution, thus allowing for optimal scaling of the
multilevel Monte Carlo Method.

`ParELAGMC` implements different sampling techniques for spatially correlated
random fields including the Karhunen–Loève expansion (KLE) for small scale problems
and stochastic PDE (SPDE) samplers for large-scale applications.
The SPDE sampler provides samples from a Gaussian random field with a Matern 
covariance function (equivalent to an exponential random field in 3D) 
and involves solving mixed finite element formulation of a 
stochastic reaction-diffusion equation with a random, white noise source 
function. Then the sampler is able to leverage existing scalable solution strategies, thus
is a scalable alternative for sampling for large-scale simulations.

Additionally, the library provides functionality to compute the Bayesian 
posterior expectation of a quantity of interest. The posterior expectation can 
be computed as a ratio of prior expectations, then approximated using Monte
Carlo sampling methods. 

The `ParELAGMC` library can support different type of deterministic problems.
In the examples, we present an application to subsurface flow simulation in
the mixed finite element setting.

Please see the following publications, and the references therein, 
for an introduction to these methods:
* [Paper 1 - SPDE sampler](https://doi.org/10.1137/16M1082688)
* [Paper 2 - SPDE sampler with non-matching mesh embedding](https://doi.org/10.1002/nla.2146)

## User Guide

### Dependencies
`ParELAGMC` requires a MPI C++ compiler, as well as the following 
external libraries:
* [ParELAG library (v2.0)](https://github.com/LLNL/parelag)
  generates the hierarchy of spatial discretizations 
  of general unstructured meshes. It builds on top of the 
  [MFEM](http://mfem.org) library for finite element methods, and the
  [hypre](http://www.llnl.gov/CASC/hypre) preconditioner and sparse numerical
  linear algebra library. `ParELAG` supports several solvers from the hypre 
  library which can be specified at runtime.

* [Tina's Random Number Generator Library (TRNG)](https://numbercrunch.de/trng/)
  is used for pseudo-random number generation which features dedicated support for
  parallel, distributed environments.
  * The library can be installed by the user **OR** the library will
     be automatically downloaded and installed.

`ParELAGMC` has optional dependencies that enable additional functionality:

* [ParMoonolith](https://bitbucket.org/zulianp/par_moonolith.git) provides 
  volume transfer of discrete fields between arbitrarily distributed 
  unstructured finite element meshes.
  * The library can be installed by the user **OR** the library will
    be automatically downloaded and installed, if enabled.

* [GLVis](http://glvis.org/) provides support for visualization of finite 
  element meshes, random field realizations, and 
  forward model problem solutions. 

### Building with CMake

[CMake](https://cmake.org/) (version 3.1 or newer) is used to generate the 
build system for `ParELAGMC`.

The CMake system maintains an infrastructure to find and properly link
the required libraries when building the `ParELAGMC` library (see the
"Dependencies" section for a complete list of the required and
optional libraries).

*PLEASE* DO NOT DO AN IN-SOURCE BUILD! It will pollute your source tree with 
various files generated and used by CMake. 
It is best practice to create a build directory.    

The configuration step is performed by running 

```bash
   mkdir <build-dir> ; cd <build-dir>                                
   cmake <source-dir> [OPTIONS] ...
```
Optionally, a shell-script can be used to set the options and invoke CMake. Using a script
provides a convenient way to repeat the same configuration repeatedly. 
Some shell-script templates are located in `cmake/example_scripts` 
that provide guidance on invoking CMake. These are provided merely as a 
suggestion for different ways to build `ParELAGMC`. 

For CMake to find the required libraries, set the environment variables 
  * `ParELAG_DIR=/path/to/parelag/build` Note: ParELAG *must* be configured using CMake. 
  
  * `TRNG_DIR=/path/to/trng/install` If not found (or not specified), 
    the library will be automatically downloaded and installed.
    `TRNG_INSTALL_PREFIX` can be used to set install location,
    otherwise it is installed in `<build-dir>/external/trng4`. 

Some important flags/options are (defaults are in [brackets]):

  * `ParELAGMC_ENABLE_ParMoonolith:BOOL={[ON],OFF}`

    * This enables the library ParMoonolith which provides volume transfer
    of discrete fields between arbitrarily distributed unstructured finite
    element meshes. This is required for the SPDE sampler using non-matching
    meshes to mitigate the artificially inflated variance.

    * If enabled, the installed library is searched for in
    `ParMoonolith_DIR`. If the library is not found, the library is automatically
    downloaded and installed. The install location can be set with
    `ParMoonolith_INSTALL_PREFIX`, other is `<build-dir>/external/par_moonolith`.

  * `ParELAGMC_BUILD_EXAMPLES:BOOL={[ON],OFF}`: Build the examples directory.

  * `ParELAGMC_BUILD_SPE10_EXAMPLES:BOOL={ON,[OFF]}`: Build the examples/SPE10 directory.

After CMake configures and generates successfully, change to your
build directory (the directory that is listed after "Build files have
been written to: " at the bottom of the CMake output) and invoke the
build system (i.e. call `make` or `ninja`, etc.) to build the library. 

The build can be tested by running `ctest` in the build directory.

[Doxygen](http://www.doxygen.org) documentation can be built by executing
`make doc` in the build directory.

### Using ParELAGMC
The intent of the library is to provide a modular methodology to specify
a Monte Carlo simulation by defining a forward model problem        
solver and a sampler strategy.  

Currently, the forward model problem is a mixed Darcy 
problem (implemented in `DarcySolver`), whereas the sampler of a log-normal spatially correlated random 
field (that is exp(s) where s is a Gaussian random field) 
can be one of the following:

* Truncated KLE with a Matern covariance function (equivalent to an exponential 
  random field in 3D) or an analytic exponential covariance.

* SPDE sampler (3 different implementations)

  * `PDESampler`: Solve SPDE on original mesh (the variance may be artificially 
    inflated along the boundary, especially near corners of spatial domain).  

  * `EmbeddedPDESampler`: Solve SPDE on an enlarged, embedded mesh (that matches 
    the original mesh) then projects the sample to the original mesh.

  * `L2ProjectionPDESampler`: Solve SPDE on an enlarged, structured 
    (non-matching) mesh then projects the sample to the original mesh. The two 
    meshes can be arbitrarily distributed among MPI processes. This is the 
    recommended sampling strategy. 

Note: For the SPDE sampler with mesh embedding, the boundary of the enlarged/embedded 
mesh should be at least a correlation length away from the boundary of the 
original mesh. 

The linear solvers for the forward model problem and the saddle point 
linear system for the SPDE sampler are specified at runtime from the software 
framework within ParELAG, using solvers and preconditioners from the HYPRE
preconditioner and sparse numerical linear algebra library.

#### Examples
The examples in the `examples/` directory build by default. Calling
one without arguments uses default values. 
To specify parameter values, a XML parameter list is specified via command 
line (`--xml-file parameter_list.xml`).  

Sample parameters are found in `build_dir/examples/example_parameters.xml` and 
other parameters lists can be found in `src_dir/examples/example_parameterlists`. 

The directory `/meshes` contains example finite element meshes, including
embedded meshes with matching and non-matching interfaces useful for running 
examples. Otherwise, a simple finite element mesh is built in the example.

Some examples include:
 * `DarcyTest.cpp`: Solve a mixed Darcy problem with a deterministic permeability coefficient. 

 * `DarcyTest_RandomInput.cpp`: Solve a mixed Darcy problem with a random 
    permeability coefficient generated with the SPDE sampler with non-matching mesh embedding.  
 
 * `EmbeddedPDESamplerTest.cpp`: Computes various statistics and realizations of the SPDE sampler with matching mesh embedding.

 * `KLSampler.cpp`: Samples a truncated KL Expansion of a random field, where underlying covariance function is either Matern or analytic exponential.

 * `MLMC.cpp`: Run a MLMC simulation for a mixed Darcy problem with random permeability realizations computed with either a truncated KLE using an analytic exponential or Matern covariance function, or the SPDE sampler (without mesh embedding).

 * `MLMC_ProjectionPDESampler.cpp`: Run a MLMC simulation for a mixed Darcy problem with random permeability realizations computed with the SPDE sampler with non-matching mesh embedding. 
 
 * `PDESamplerTest.cpp`: Computes various statistics and realizations of the SPDE sampler without mesh embedding.

 * `ProjectionPDESamplerTest.cpp`: Computes various statistics and realizations of the SPDE sampler with non-matching mesh embedding. 

 * `SLMC.cpp`: Single-level Monte Carlo simulation for a mixed Darcy problem with a random 
permeability coefficient samples using either a KL expansion (analytic exponential or Matern covariance function) or SPDE sampler with (non-matching) mesh embedding.

Additionally, the library provides functionality to compute the Bayesian 
posterior expectation of a quantity of interest. The posterior expectation can 
be computed as a ratio of prior expectations using Bayes' rule, then 
approximated using Monte Carlo sampling methods. That is, 
E_posterior[Q] = E[Q \Pi_{likelihood}]/E[\Pi_likelihood]. 
Also, the splitting estimator can be computed, that is,
Splitting-E_posterior[Q] = E[(Q \Pi_likelihood)/(\Pi_likelihood)].

This methodology is examined in the following examples:
 
 * `RatioEstimator_MC.cpp`: Computes ratio estimates using single-level MC estimators for a fixed number of samples. 
 
 * `RatioEstimator_MC_Manager.cpp`: Computes the ratio estimate (either splitting or standard) using single-level MC estimators where the number of samples is determined 'on the fly'. 

 * `RatioEstimator_MLMC.cpp`: Computes ratio estimates using MLMC estimators for a fixed number of samples.

 * `RatioEstimator_MLMC_Manager.cpp`: Computes the ratio estimator (either splitting or standard) using MLMC estimators where the number of samples is determined 'on the fly'.

Examples ending with `_Legacy` indicate that the forward model problem 
and sampler use a particular solver/preconditioner strategy to solve the 
resulting saddle point problems 
as specified in this [publication](https://doi.org/10.1137/16M1082688). 

An optional directory of examples using the permeability data from the 
[SPE Comparative Solution Project Model 2](www.spe.org/web/csp/dataset/set02.htm) 
are found in `examples/SPE10`.

### License 
Copyright information and licensing restrictions can be found in the file 
`COPYRIGHT`. The `ParELAGMC` library is licensed under the GPL v2.0, 
see the file `LICENSE`. 

