# CSC3034-ANN

Quick setup and run guide for teammates.

## Prerequisites
- Windows with PowerShell
- Python 3.13 (or 3.11/3.12 compatible)

## 1) Clone and enter the project folder
```powershell
git clone <repo-url>
cd CSC3034-ANN
```

## 2) Create and activate a virtual environment
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```
If you see a policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
then activate again.

## 3) Install dependencies
```powershell
pip install -r requirements.txt
```

## 4) Prepare the dataset
- The CSV is provided at `Dataset/Exam_Score_Prediction.csv`.
- The current script `main.py` expects the CSV at the repo root with name `Exam_Score_Prediction.csv`.

Options:
- Option A (copy file to root):
	```powershell
	Copy-Item Dataset/Exam_Score_Prediction.csv ./Exam_Score_Prediction.csv
	```
- Option B (run with path arg):
	```powershell
	python ./main.py Dataset/Exam_Score_Prediction.csv
	```

## 5) Run the model
```powershell
python ./main.py
```
Outputs are saved to `Output/`:
- `loss_curve.png`: training vs validation loss
- `true_vs_pred.png`: scatter of true vs predicted scores

## Troubleshooting
- Activation script not found: recreate the venv
	```powershell
	Remove-Item -Recurse -Force venv; python -m venv venv
	```
- TensorFlow install issues: ensure you’re on a 64-bit Python and recent pip. Update pip:
	```powershell
	python -m pip install --upgrade pip
	```

## Project files
- `main.py`: trains and evaluates a Keras model
- `dataset_prep.py`: dataset utilities (if used)
- `requirements.txt`: Python dependencies
- `Dataset/Exam_Score_Prediction.csv`: input data
- `Output/`: generated plots