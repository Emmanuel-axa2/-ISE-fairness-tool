# ISE Fairness Tool — Boundary-Focused Sampling for AI Model Fairness Testing

## Overview
This repository contains two tools for automated AI model fairness testing developed as part of the Intelligent Software Engineering coursework at the University of Birmingham.

- `lab4_solution.py` — Baseline: Random Search
- `solution.py` — Our Tool: Boundary-Focused Sampling (BFS)

Both tools find Individual Discriminatory Instances (IDIs) across 8 real-world datasets. An IDI is a pair of inputs identical except for a sensitive feature (e.g. race, gender) that receives different predictions from the model.

## Setup

1. Clone the repository:
 git clone https://github.com/Emmanuel-axa2/-ISE-fairness-tool.git
cd -ISE-fairness-tool
2. Create and activate a virtual environment:
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
3. Install dependencies: 
pip install tensorflow-macos numpy pandas scikit-learn
4. Download datasets and models from the lab GitHub and place them in the correct folders:
https://github.com/ideas-labo/ISE/tree/main/lab4
5.## Running the Tools
Run the baseline:
python lab4_solution.py
Run our solution:
python solution.py
## Running the Tools

File Structure:
├── dataset/              # CSV datasets
├── DNN/                  # Pretrained .h5 models
├── lab4_solution.py      # Baseline Random Search
├── solution.py           # Boundary-Focused Sampling
├── results/              # Output CSV results
├── requirements.pdf      # Dependencies and installation
├── manual.pdf            # Usage manual
└── replication.pdf       # Replication instructions

## Documentation
For full details see the PDF files in the root directory:
- `requirements.pdf` — Dependencies and installation
- `manual.pdf` — How to use the tools
- `replication.pdf` — How to replicate reported results