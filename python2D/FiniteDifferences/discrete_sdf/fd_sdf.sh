command="python normalder.py --iter 256 --distance 0.01"
eval "$command"
command="python normalder.py --iter 64--distance 0.1"
eval "$command"
command="python normalder.py --iter 32 --distance 0.2"
eval "$command"


command="python grad.py --iternormal 256 --distance 0.01"
eval "$command"
command="python grad.py  --iternormal 64 --distance 0.1"
eval "$command"
command="python grad.py  --iternormal 32 --distance 0.2"
eval "$command"


#command="python fd.py --fdstep 5e-3"
#eval "$command"

#command="python fd.py --fdstep 1e-3"
#eval "$command"
