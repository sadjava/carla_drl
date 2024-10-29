# Carla DRL

Carla DRL is a project that uses Deep Reinforcement Learning (DRL) to train an agent to drive a car in the Carla simulator.

## Project structure
```
Carla-DRL/
├── .gitignore                  # Specifies files and directories to be ignored by Git.
├── .coveragerc                 # Configuration for coverage.py, specifying files to omit from coverage reports.
├── .mypy.ini                   # Configuration for mypy, ignoring missing imports for certain libraries.
├── .pylintrc                   # Configuration for Pylint, checking the quality of Python code.
├── pyproject.toml              # Configuration for Poetry, managing dependencies and project metadata.
├── pytest.ini                  # Configuration for pytest, specifying test paths and options.
├── README.md                   # Main documentation file providing an overview and instructions.
├── measure_speed.py            # Script for measuring the speed of the agent in the simulation.
├── carla_drl/                  # Main package directory for the Carla DRL project.
│   ├── __init__.py             # Marks the directory as a Python package.
│   ├── depth_estimation/       # Contains files related to depth estimation tasks.
│   │   ├── __init__.py         # Marks the directory as a Python package.
│   │   ├── dataset.py          # Dataset class for loading depth estimation data.
│   │   ├── midas.py            # Implements the MonoDepthNet model for monocular depth estimation.
│   │   ├── train.py            # Training logic for the depth estimation model.
│   │   └── utils.py            # Utility functions for depth estimation tasks.
│   ├── lane_following/         # Contains files related to lane following tasks.
│   │   ├── agent.py            # Implements the agent for lane following.
│   │   ├── encoder.py          # Contains the encoder for processing inputs.
│   │   ├── model.py            # Defines the model architecture for lane following.
│   │   ├── parameters.py       # Configuration parameters for the lane following model.
│   │   └── train.py            # Training logic for the lane following model.
│   └── semantic_segmentation/  # Contains files related to semantic segmentation tasks.
│       ├── __init__.py         # Marks the directory as a Python package.
│       ├── dataset.py          # Dataset class for loading semantic segmentation data.
│       ├── train.py            # Training logic for the semantic segmentation model.
│       ├── unet.py             # Implements the UNet model for semantic segmentation.
│       └── utils.py            # Utility functions for semantic segmentation tasks.
├── simulation/                 # Contains files related to the simulation environment.
│   ├── __init__.py             # Marks the directory as a Python package.
│   ├── connection.py           # Manages the connection to the Carla simulator.
│   ├── environment.py          # Defines the environment for the simulation, including vehicle and pedestrian management.
│   ├── sensors.py              # Contains the camera sensor configuration and management for the simulation.
│   ├── settings.py             # Configuration parameters for the simulation.
│   └── sync_data_generation.py # Script for generating synchronized data from the Carla simulator.
└── .github/                    # Contains GitHub Actions workflows for CI/CD.
    └── workflows/              # Directory for workflow files.
        └── python-tests.yml    # Defines the workflow for running Python tests on push and pull request events.
```
