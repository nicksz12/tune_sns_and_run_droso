# tune_sns_and_run_droso

To get started:
- Run neural_motor_control_demo.py.
- This will open a MuJoCo visualizer and show the model as the simulation progresses.
- When the simulation ends, every neural and mechanical state will be saved into a .json.
- The plot_data function will automatically run and prompt the user to select data to plot.
-- This may not work properly, in which case you can paste the name of the .json into the test_plot_data.py script.
-- You could visualize the data any way you wish! plot_data is meant to be the bare minimum.

To-do:
- Complete intra-leg coordination pathways.
- Integrate body plan and kinematics to tun descending pathways.
- Integrate load feedback.
- Interface with Drosophibot's I/O.
