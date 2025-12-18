# ECE 595 Reinforcement Learning

## **Physics Informed Control Recovery for Race Cars**

## 1. Overview

This instruction provides guidance on installing the Assetto Corsa game simulation and setting up an environment to train and experiment with a model that recovers the car when it slips inside the game.

Features:
- High-Fidelity Simulation: Realistic car dynamics and track environments.
- Customizable Scenarios: Various cars, tracks, and weather conditions.
- RL Integration: Compatible with Gym for easy application of RL algorithms.
---

## 2. System Requirements
The hardware specifications below are the specs we used to set up the simulation and train the reinforcement learning model.
### 2.1 Hardware
- CPU: Intel i7-13700K
- GPU: RTX A4000
- RAM: > 16 GB
- Storage: > 200 GB

### 2.2 Software
- OS: Windows 11
- Python: 3.9
- CUDA: 12.8
- Simulator: Assetto Corsa (Steam)

---

## 3. Repository Structure

```text
project_root/
├── assetto_corsa_gym/           # Assetto Corsa simulation environment
│   ├── AssettoCorsaEnv/         # Core environment implementation
│   │   ├── ac_env.py            
│   │   ├── recovery_ac_env.py   
│   │   └── ac_client.py         
│   ├── AssettoCorsaConfigs/     # Vehicle and track configurations
│   │   ├── cars/                
│   │   └── tracks/              
│   └── AssettoCorsaPlugin/      # Python plugin for sensor data and telemetry
│       └── plugins/sensors_par  
├── algorithm/                   # RL algorithms
│   └── discor/                  
│       └── discor/
│           ├── agent.py
│           ├── network.py       
│           |── replay_buffer.py 
|           └── algorithm/
|               ├── base.py
|               ├── ddpg.py
|               ├── discor.py
|               ├── eval.py
|               ├── sac.py
|               └── td3.py
├── common/                      # Shared utilities
│   ├── logger.py                
│   └── misc.py                  
├── outputs_recovery/            # Training outputs and checkpoints
├── train_recovery.py            # Recovery-specific training
├── test_recovery.py             # Recovery-specific testing
├── config.yml                   # Main configuration file
├── ac_offline_train_paths.yml   
└── requirements.txt             
```

## 4. Environment Setup
### 4.1 Python Environment
**Install Visual Studio Compiler**
- To compile the necessary components for the plugin, download and install the Visual Studio compiler from: [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Make sure to install the C++ build tools component

**Install Python using Anaconda**
```bash
conda create -n p309 python=3.9.13
conda activate p309
pip install setuptools==65.5.0 "cython<3"
pip install "wheel<0.40.0"
python -m pip install pip
pip install -r requirements.txt
conda install pytorch==1.12.1 cudatoolkit=11.6 \
    -c pytorch -c conda-forge
```

## 5. Simulation - Assetto Corsa Gym
### 5.1 Steam Installation
- Download Steam from the following webpage. (link: https://store.steampowered.com/about/)
- After installing Steam, purchase Assetto Corsa. (Note: Game purchase cost applies)
- After purchasing the game, proceed with installation on your local environment.

### 5.2 Assetto Corsa Plugin Installation

**Install vJoy**  
vJoy is required to send commands to Assetto Corsa.  

🔗 **[Download and install vJoy](https://sourceforge.net/projects/vjoystick/)**  

<br>

**Copy the Plugin Files**  
1. Locate the plugin folder in this repository:  
   ```bash
   .\assetto_corsa_gym\AssettoCorsaPlugin\
       \plugins\sensors_par
   ```
2. Copy this folder to the Assetto Corsa installation directory under `apps\python\`.  
   - The default AC installation path is:  
     ```bash
     C:\Program Files (x86)\Steam\steamapps\
         \common\assettocorsa\
     ```
3. The final destination should look like this:  
   ```bash
   C:\Program Files (x86)\Steam\steamapps\
       \common\assettocorsa\apps\python\sensor_par
   ```

<br>

**Install Configuration Files**  
Copy the necessary configuration files from:  
```bash
assetto_corsa_gym\AssettoCorsaPlugin\
    \windows-libs
```

- VJoy Configuration
  - Copy `Vjoy2.ini` to:  
    ```bash
    C:\Users\%user%\Documents\Assetto Corsa\
        \cfg\controllers\savedsetups
    ```
- WASD Controls
  - Copy `WASD.ini` to:  
    ```bash
    C:\Users\%user%\Documents\Assetto Corsa\
        \cfg\controllers\savedsetups
    ```
- DLLs and Library Folders 
  - Copy these folders (Python socket library) to:  
    ```bash
    C:\Program Files (x86)\Steam\steamapps\
        \common\assettocorsa\system\x64
    ```

<br>

**Install Custom Shaders Patch**  
The **Custom Shaders Patch** is required to restart the car.  

1. **Download and install Content Manager**:  
   🔗 [Content Manager Download](https://acstuff.ru/app/)  
2. Open **Content Manager** → Navigate to:  
   ```bash
   Settings > Custom Shaders Patch
   ```
3. Click **Install** to complete the setup.

**<span style="color:red">Important!</span>**
**<span>Subsequent simulation runs (game execution) must be done through Content Manager to launch Assetto Corsa!!!</span>**
<br>

## 6. Configuring Assetto Corsa

**Enable the Plugin**
1. Open **Assetto Corsa**  
2. Go to **Options > General > UI Modules**  
3. Enable `sensor_par`  

**Configure Controls**
1. Navigate to **Options > Controls**  
2. Ensure that both **vJoy** and **WASD** are available  
3. Load **vJoy** as the active input  

**Set Video and Display Settings**
- **Frame Rate Limit:** `50 FPS`  
  - *(Located in `Options > Video > Display > Framerate Limit`)*  

**Start a Hotlap Session**
- **Mode:** `Challenge > Hotlap`  

**Adjust Driving Assists**
| Setting                   | Recommended Value |
|---------------------------|------------------|
| Automatic Gearbox         | **ON** |
| Ideal Racing Line         | **As Preferred** |
| Automatic Clutch          | **Enabled** |
| Automatic Throttle Blip   | **Disabled** |
| Traction Control          | **OFF** |
| Stability Control         | **OFF** |
| Mechanical Damage         | **OFF** |
| Tyre Blankets             | **ON** |
| ABS                       | **OFF** |
| Fuel Consumption          | **OFF** |
| Tyre Wear                 | **OFF** |
| Slipstream Effect         | **1x** |
| Time of Day               | **10:00 AM** |
| Weather                  | **Mid Clear** |
| Ambient Temperature       | **26°C** |
| Time Multiplier           | **1x** |
| Track Surface             | **Random** |
| Penalties                 | **ON** |

<br>

## 7. Train Model
To run the model, place the pre-defined racing line in the following location. [Download Link](https://drive.google.com/file/d/12f5PuA98XcDN8y7Rg9i519Y5v4mL32cu/view?usp=drive_link)

```
assetto_corsa_gym\AssettoCorsaConfigs\tracks\monza_0.1m.pkl
```
### 7.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Track | Monza |
| Car | BMW Z4 GT3 |
| Learning Rate | 0.0003 |
| Batch Size | 128 |
| Replay Buffer | 60,000 |
| Discount Factor (γ) | 0.992 |
| Target Update Coefficient | 0.005 |

All hyperparameters can be modified in `config.yml`.



### 7.2 Training New Model

To train a new recovery model from scratch, use the following command:

```bash
python ./train_recovery.py --algo td3 \
    --randomize_speed --randomize_steer \
    --num_episodes 10000
```

**Key Arguments:**
- `--algo`: Algorithm to use (`sac`, `td3`, `ddpg`, default: `td3`)
- `--pirl`: Enable physics reward for training
- `--randomize_speed`: Enable target speed randomization during training
- `--randomize_steer`: Enable steering intensity randomization
- `--num_episodes`: Number of training episodes (default: 10000)
- `--seed`: Random seed (default: 42)

**Example Commands:**

Train with SAC algorithm:
```bash
python ./train_recovery.py --algo sac --randomize_speed
```

Train with TD3 and full randomization:
```bash
python ./train_recovery.py --algo td3 \
    --randomize_speed --randomize_steer \
    --num_episodes 5000
```

Checkpoints are automatically saved in:
```bash
outputs_recovery/monza/[timestamp]/
```

### 7.3 Running Pre-trained Model

To evaluate a pre-trained model:
```bash
python ./test_recovery.py \
    --load_path outputs_recovery/monza/[timestamp]/model_final
```

### 7.4 Pre-trained Models

You can download our pre-trained models here:

- SAC: [Download](https://app.box.com/s/w7n028tym8z5uez1o5te152epe4hj3ly)
- TD3_non_physics: [Download](https://app.box.com/s/23227ulctimw0jzyztopj7kdznt8z1r0)
- TD3_physics: [Download](https://app.box.com/s/qdsppvts7e3n8m4sp5cgw7gulgy7zct8)