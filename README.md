# Combating Galamsey- A Hybrid Deep Learning Framework with Physics-Informed Post-Processing for Real-Time Acoustic Detection of Illegal Mining Activities

This is a robust, end-to-end acoustic surveillance framework designed to combat illegal artisanal mining ("Galamsey") in Ghanaian forest. By leveraging a Hybrid Dual-Stream Deep Learning architecture and Physics-Aware Logic, this system detects heavy machinery (excavators, chainsaws) in real-time and dispatches geo-tagged alerts to stakeholders via Telegram. Key FeaturesHybrid Feature Fusion: Combines Log-Mel Spectrograms (Texture) and MFCCs (Timbre) using a weighted ensemble CNN to achieve 98.68% accuracy.Physics-Aware Logic Check: A novel deterministic layer that uses Spectral Band Power Ratios (Bass vs. Treble) to validate AI predictions, eliminating false positives caused by environmental noise.Real-Time Inference: Optimized for edge deployment with a 1.5-second sliding window and adaptive noise gating.IoT Alerting: Integrated Simulink-to-Telegram pipeline that pushes instant "Threat Detected" notifications with Google Maps GPS coordinates. Repository StructureThe project is organized into modular scripts for reproducibility
:Bash
├── data_raw/                  # (Not included) Place your .wav dataset here
|  ├── background/             # background_1.wav
|  ├── chainsaw/               # chainsaw_1.wav
|  ├── engine/                 # engine_1.wav
├── data_processed/            # Generated features (Mel/MFCC matrices)
├── manifests/                 # CSV files for Train/Val/Test splits
├── results/                   # Trained models (.mat) and performance metrics
├── figures/                   # Generated plots for the manuscript
├── scripts/
│   ├── Module_1_Preprocessing.m   # Feature Extraction & Augmentation
│   ├── Module_2_Training.m        # Dual-Stream CNN Training
│   ├── Module_3_Ensemble.m        # Late Fusion & Evaluation
│   ├── Module_4_Visualization.m   # t-SNE & Waveform Plots
│   ├── Module_5_Inference.m       # Standalone Inference Script
|   ├── Module_6_Data_Simulink     # Prepare Models for Simulink 
│   ├── run_galamsey_inference.m   # Helper function for Simulink
│   └── send_telegram_alert.m      # IoT Alerting Logic
└── README.md

Installation & SetupPrerequisitesMATLAB R2021a or laterDeep Learning ToolboxAudio ToolboxSimulink (for real-time deployment simulation)1. Clone the RepositoryBashgit clone https://github.com/jakabs/Real-Time-Acoustic-Detection-of-Illegal-Mining-Activities-in-Ghanain-Forest/

2. Prepare the DatasetPlace your raw .wav files in the data_raw/ folder, organized by class:data_raw/background/data_raw/chainsaw/data_raw/engine/3. Run the PipelineExecute the modules in order from MATLAB:Module_1_Preprocessing.m: Extracts features and creates manifests.Module_2_Training.m: Trains the Mel and MFCC networks.Module_3_Ensemble.m: Optimizes weights and evaluates accuracy. Real-Time Deployment (Simulink)To run the live detection prototype:Open MATLAB and navigate to scripts/.Run the Preparation Script to pack models into simulink_data_package.mat.Open the Simulink model file (not included, but easy to recreate using the run_galamsey_inference block).Configure Telegram:Open send_telegram_alert.m.Replace bot_token and chat_id with your credentials.Click Run in Simulink. The system will now listen to your microphone and alert you upon detecting threats. PerformanceModel ArchitectureInput FeatureTest AccuracyPrimary StreamLog-Mel Spectrogram98.21%Secondary StreamMFCC96.45%Hybrid EnsembleFused (Mel + MFCC)98.68%Confusion Matrix and t-SNE plots are available in the figures/ directory. Citation If you use this code in your research, please cite:Code snippet@article{Okoampah2025,
  title={Combating Galamsey- A Hybrid Deep Learning Framework with Physics-Informed Post-Processing for Real-Time Acoustic Detection of Illegal Mining Activitiesm},
  author={Okoampah, Ernest},
  journal={ },
  year={2025}
This project is licensed under GPL-3.0 License (General Public License version 3)- see the LICENSE file for details.Author: Ernest Okoampah Contact: ernestosei36@yahoo.com
