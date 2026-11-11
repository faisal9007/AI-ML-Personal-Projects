# Bangla License Plate Reader (VS Code project)

Two YOLO models chained together:

1. **Plate detector** finds the plate on a vehicle photo (`Vehicle_to_License_Plate` data).
2. **Character reader** finds every character on the cropped plate (`License_Plate_to_Text` data, 102 classes). Sorting the boxes gives the plate text, including two-line plates.

## Files

| File | What it does |
|---|---|
| `config.py` | paths and settings (epochs, image sizes, model sizes) |
| `textlogic.py` | label reading, character ordering, edit distance |
| `prepare_data.py` | finds or downloads the dataset, fixes the YOLO yaml files |
| `explore.py` | sample pictures and class balance, saved to `reports/` |
| `train.py` | trains the models, copies the results to `models/` |
| `evaluate.py` | mAP, exact-match accuracy, best threshold, error pictures, saved to `reports/` |
| `plate_reader.py` | the two-stage pipeline used by everything below |
| `predict.py` | read plates from an image or folder |
| `video.py` | read plates from a video and log them with timestamps |
| `streamlit_app.py` | web app (best base for your own UI design) |
| `gradio_app.py` | quick demo |
| `.vscode/launch.json` | Run and Debug entries for every step |

## Setup (PowerShell)

Open this folder in VS Code (File → Open Folder, or add it to your workspace). Select your interpreter with Ctrl+Shift+P → **Python: Select Interpreter** and choose your `.venv`.

```powershell
cd "D:\Python\Personal Projects\Bangla\plate_reader"      # wherever you put this folder
& "D:\Python\Personal Projects\.venv\Scripts\Activate.ps1"
python -c "import torch; print(torch.cuda.is_available())"  # skip if torch is not installed yet
```

**GPU:** if you have an NVIDIA GPU and the line above prints `False`, install the CUDA build of PyTorch first. Get the exact command for your CUDA version from https://pytorch.org/get-started/locally/ (it looks like `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`). Then:

```powershell
pip install -r requirements.txt
```

**No GPU?** Do not train locally. The character reader would take many hours on CPU. Train with the Colab notebook, then copy `plate_detector.pt`, `char_reader.pt` and `char_names.json` into the `models/` folder here. Everything from step 4 onward works on CPU.

## Run

```powershell
python prepare_data.py          # 1. dataset + fixed configs
python explore.py               # 2. optional, pictures in reports/
python train.py --model both    # 3. train (use --model plate or --model char for one)
python evaluate.py              # 4. metrics in reports/
python predict.py "path\to\car.jpg"
streamlit run streamlit_app.py
```

Or use the Run and Debug panel (Ctrl+Shift+D) and pick a step from the dropdown.

If Windows training crashes inside the dataloader, set `WORKERS = 0` in `config.py`.
If VS Code or the training stops midway, run `python train.py --model char --resume`.

## Using the dataset you already downloaded

If this folder sits inside `Bangla\` (next to `data\license_plate`), it is found automatically. Otherwise point to it:

```powershell
$env:PLATE_DATA_DIR = "D:\Python\Personal Projects\Bangla\data\license_plate"
```

Otherwise `prepare_data.py` downloads it into `data\license_plate` here.

## Notes

- Horizontal flip is disabled for the character reader on purpose: a mirrored Bengali character is a different character.
- The picture shows English transliteration (OpenCV cannot draw Bengali). The Bengali text appears in the app's text output.
- The dataset's test split is small (166 vehicle photos, 252 plate crops), so metrics move between runs. Report validation numbers next to test numbers.
- `data/`, `runs/`, `reports/`, `outputs/` and `models/*.pt` are in `.gitignore`.

## Citation (required by the dataset license, CC BY 4.0)

Dataset: Bangla LPDB-A (YOLO-ready version by Ashikur Rahman Shad, Kaggle), originally by Ataher Sams and Homaira Huda Shomee (Zenodo, 2021, https://doi.org/10.5281/zenodo.4718238).

H. H. Shomee and A. Sams, "License Plate Detection and Recognition System for All Types of Bangladeshi Vehicles Using Multi-step Deep Learning Model," 2021 Digital Image Computing: Techniques and Applications (DICTA), Gold Coast, Australia, pp. 1-7, doi:10.1109/DICTA52665.2021.9647284.
