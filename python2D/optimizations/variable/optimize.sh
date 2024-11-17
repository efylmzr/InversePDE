command="python run_variable.py --iternum 2048 --scale 3 --coeff source --screening 0 --conf 1 --restensor 16" 
eval "$command"
command="python run_variable.py --iternum 2048 --scale 3 --coeff source --screening 10 --conf 2 --restensor 16"
eval "$command"


command="python run_variable.py --iternum 2048 --scale 3 --coeff screening --conf 1  --bias 1.0 --restensor 16"
eval "$command"
command="python run_variable.py --iternum 2048 --scale 3 --coeff screening --conf 2 --restensor 16"
eval "$command"


command="python run_variable.py --iternum 1024 --scale 3 --coeff diffusion --screening 0 --conf 1 --restensor 16 --bias 1.0 --dirichlet"
eval "$command"
command="python run_variable.py --iternum 1024 --scale 3 --coeff diffusion --screening 10 --conf 2 --restensor 16 --bias 1.0 --dirichlet"
eval "$command"