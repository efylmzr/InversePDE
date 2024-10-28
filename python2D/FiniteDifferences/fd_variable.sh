
fdstep=("0.001")
res=("64")
spp=1024
for fd in "${fdstep[@]}"
    do
    for r in "${res[@]}"
    do
    #command="python eit_variable/eit_fd.py --restensor $r --fdstep $fd"
    #eval "$command"
    command="python variable/fd_source.py --restensor $r --fdstep $fd"
    eval "$command"
    command="python variable/fd_screening.py --restensor $r --fdstep $fd"
    eval "$command"
    command="python variable/fd_diffusion.py --restensor $r --fdstep $fd"
    eval "$command"
    done
done
