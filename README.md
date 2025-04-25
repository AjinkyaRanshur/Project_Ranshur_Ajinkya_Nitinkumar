# Project_Ranshur_Ajinkya_Nitinkumar
This repository is for the assignment for the course called Image and Video Processing by Chaitanya Guttikar


/
├── __pycache__/                 # Python bytecode cache  
├── data/                        # Expected CIFAR‑100 train/test folders  
├── plots/results/               # Saved adversarial plots  
├── cmd_list                     # Single-line job command: interface.py with train+attack   
├── cmd_list_test                # Command to check CUDA availability 
├── checkcuda.py                 # Prints torch.cuda.is_available()    
├── config.py                    # Hyperparameters & paths 
├── dataset.py                   # DataLoader setup for CIFAR‑100   
├── interface.py                 # CLI tying model/train/predict/adversarial tests  
├── model.py                     # Custom VGG‑16 architecture implementation   
├── predict.py                   # Single‑image inference function  
├── run.sh                       # SLURM submission script  
├── train.py                     # Training loop, checkpointing & plotting   
├── training_plot.png            # Example loss/accuracy plot  
└── README.md                    # Project description
