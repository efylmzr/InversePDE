#command="python normalder.py --distance 0.01"
#eval "$command"
#command="python normalder.py --distance 0.1"
#eval "$command"
#command="python normalder.py --distance 0.2"
#eval "$command"


command="python grad.py --distance 0.01"
eval "$command"
command="python grad.py --distance 0.1"
eval "$command"
command="python grad.py --distance 0.2"
eval "$command"


#command="python fd.py --fdstep 5e-3"
#eval "$command"

