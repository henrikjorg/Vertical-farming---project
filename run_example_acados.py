import sys
import os

# Add the directory to sys.path
script_dir = os.path.join('acados', 'examples', 'acados_python', 'getting_started')
full_script_dir = os.path.join(sys.path[0], script_dir)  # Assuming relative to the script's directory
sys.path.append(full_script_dir)

# Now you can import normally
from minimal_example_closed_loop import main as closed_loop_main  # Assuming there's a main function
from minimal_example_ocp import main as ocp_main # Assuming there's a main function
from minimal_example_sim import main  as sim_main # Assuming there's a main function

# Remove the directory from sys.path after import if needed
sys.path.remove(full_script_dir)

# Use the imported function/class
#closed_loop_main()
ocp_main()
#sim_main()
