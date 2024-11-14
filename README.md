# InversePDE

**InversePDE** is a repository containing the implementations of experiments from the paper _"Solving Inverse PDE Problems using Grid-Free Monte Carlo Estimators"_ by **Ekrem Fatih Yilmazer, Delio Vicini, and Wenzel Jakob**.

## Repository Structure

The repository is organized as follows:

- **\`PDE2D\`** and **\`PDE3D\`**: Source files for 2D and 3D solvers, respectively.
- **\`python2D\`** and **\`python3D\`**: Contains optimization scripts and validation experiments.
- **\`notebooks-2D\`** and **\`notebooks-3D\`**: Jupyter notebooks for visualizing various tests and generating results.

## Package Details

- **3D Solver**: 
  - Requires Signed Distance Function (SDF) representations for shapes or spheres.
  - Currently supports Dirichlet boundary conditions only.

- **2D Solver**: 
  - Supports representations using Quadratic Bézier Curves, SDFs, and Circles.
  - Handles both Neumann and Dirichlet boundary conditions.

## Running Experiments

1. **Generate Results**:  
   Execute the shell scripts located in the \`python2D\` and \`python3D\` directories to reproduce the experimental results presented in the paper. 3D results require generation of a high resolution SDF from a mesh, you 
   can simply run \`redistance/run.py\ for generation of the SDF. 
   
2. **Generate Figures**:  
   After running the experiments, use the Jupyter notebooks located in \`notebooks-2D/figure-generations\` and \`notebooks-3D/figure-generations\` to generate figures based on the computed results.
