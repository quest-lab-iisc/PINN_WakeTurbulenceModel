# A PINN for Turbulent Simulations

A Physics-informed neural network model to simulate turbulent flows by training only on PDE-residual loss of RANS equations and problem-specific boundary data.
The developed model is validated by comparing the PINN solution with numerical solvers and analytical solutions for three test cases:

1. Turbulent channel flow
    - Re = 180, 395 and 590
2. ABL flow
3. Turbulent wake behind a single wind turbine
    - Sexbierum and Nibe wind turbines
    - Transfer learning

- run pip install -r requirements.txt file before executing the python scripts
---------------------------------------------------------------------------
## 1. Turbulent channel flow
1. Plots are saved in Turbulent_channel_flow/Plots folder
2. Trained models are saved in Turbulent_channel_flow/Re_^/Saved_model (^ - 180, 395, 590)

    First run the below python script
    - python3 Turbulent_channel_flow/Re_^/Run_channel_flow.py (^ - 180, 395, 590)

    Run all cells in the below python notebook to get plots
    - Turbulent_channel_flow/Post_Processing.ipynb
---------------------------------------------------------------------------
## 2. ABL model
1. Plots are saved in ABL_flow/Plots folder.
2. Trained models are saved in ABL_flow/Saved_model

    First run the below python script
    - python3 ABL_flow/Run_ABL_model.py

    Run all cells in the below python notebook to get plots
    - ABL_flow/Post_Processing.ipynb
---------------------------------------------------------------------------
## 3. Turbulent wake behind a single wind turbine
1. Plots are saved in Turbulent_wake_simulation/^_case/Plots folder (^ - Sexbierum, Nibe)
2. Trained models are saved in Turbulent_wake_simulation/^_case/Saved_model  (^ - Sexbierum, Nibe)
3. Data for generating contour plots are saved in Turbulent_wake_simulation/^_case/^_contour_data.csv (^ - Sexbierum, Nibe)

    First run the below python script
    - python3 Turbulent_wake_simulation/^_case/Run_wake_model.py (^ - Sexbierum, Nibe)

    Run all cells in the below python notebook to get plots
    - Turbulent_wake_simulation/^_case/Post_Processing.ipynb (^ - Sexbierum, Nibe)
 
### Transfer Learning
1. Plots are saved in Turbulent_wake_simulation/Transfer_learning/Plots folder
2. Reference model is saved in Turbulent_wake_simulation/Transfer_learning/Reference_model
3. Pre-trained model is saved in Turbulent_wake_simulation/Transfer_learning/Pretrained_model
4. Model after transfer learning is saved in Turbulent_wake_simulation/Transfer_learning/Transfer_learned_model
 
    First run the below python script
    - python3 Turbulent_wake_simulation/Transfer_learning/Run_wake_model_TL.py
    
    Run all cells in the below python notebook to get plots
    - Turbulent_wake_simulation/Transfer_learning/Post_Processing.ipynb
 
    
