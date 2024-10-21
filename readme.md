# A PINN for Turbulent Wake Simulations

A Physics-informed neural network model to simulate turbulent flows by training only on PDE-residual loss of RANS equations and problem-specific boundary data.

- run pip install -r requirements.txt file before executing the python scripts
---------------------------------------------------------------------------
## 3D Turbulent wake behind a single wind turbine
- Sexbierum and Nibe wind turbines

- Plots are saved in Turbulent_wake_simulation/^_case/Plots folder (^ - Sexbierum, Nibe)
- Trained models are saved in Turbulent_wake_simulation/^_case/Saved_model  (^ - Sexbierum, Nibe)
- Data for generating contour plots are saved in Turbulent_wake_simulation/^_case/^_contour_data.csv (^ - Sexbierum, Nibe)

    First run the below python script
    1. python3 Turbulent_wake_simulation/^_case/Run_wake_model.py (^ - Sexbierum, Nibe)

    Run all cells in the below python notebook to get plots
    1. Turbulent_wake_simulation/^_case/Post_Processing.ipynb (^ - Sexbierum, Nibe)
---------------------------------------------------------------------------
## 2D cases
1. Turbulent channel flow
    - Re = 180, 395 and 590
2. Velocity driven flow

### 1. Turbulent channel flow
- Plots are saved in Turbulent_channel_flow/Plots folder
- Trained models are saved in Turbulent_channel_flow/Re_^/Saved_model (^ - 180, 395, 590)
- Matlab code for RANS k-epsilon model is in RANS_data/RANS_matlab

    First run the below python script
    1. python3 Turbulent_channel_flow/Re_^/Run_channel_flow.py (^ - 180, 395, 590)

    Run all cells in the below python notebook to get plots
    1. Turbulent_channel_flow/Post_Processing.ipynb

### 2. Velocity driven flow
- Plots are saved in Velocity_driven_flow/Plots folder.
- Trained models are saved in Velocity_driven_flow/Saved_model

    First run the below python script
    1. python3 Velocity_driven_flow/Run_Velocity_driven_model.py

    Run all cells in the below python notebook to get plots
    1. Velocity_driven_flow/Post_Processing.ipynb
