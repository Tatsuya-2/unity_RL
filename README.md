<!-- GETTING STARTED -->
## Getting Started
### Prerequisites

- Conda
  Anaconda or Miniforge3 or ...

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Tatsuya-2/unity_RL.git
   ```
2. Create environments
   ```sh
   cd unity_RL
   conda env create --file myenv.yaml
   ```
3. Activate environments
   ```sh
   conda activate ml_unity_baselines
   ```

<!-- USAGE EXAMPLES -->
## Usage
1. Run hello mlagents.
   ```sh
   python hello_mlagents.py
   ```
2. Train agents. It takes few hours.
   ```sh
   python train_mlagents.py
   ```
3. Predict agents.
   ```sh
   python predict_mlagents.py
   ```