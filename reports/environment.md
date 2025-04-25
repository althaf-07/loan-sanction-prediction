# 🛠️ Environment Setup & Reproducibility

## ⚙️ Tech Stack & System Info

- **Hardware**: No hardware limitations. This project is lightweight and runnable on any modern machine, including local setups, Colab, and Kaggle.
- **OS**: This project was developed on an Ubuntu-based machine. I tried to maintain this project OS independent. But there is no guarantee that it is. Just a small heads-up.
- **Python Version**: 3.12
- **Python Dependency Manager**: `uv` (v0.6.16)
- **Project Configuration and Dependencies**: Listed in [`pyproject.toml`](../pyproject.toml)

---

## Setup Guides

### Ubuntu-based PC

#### 1. 📦 Install Required System Packages

```bash
sudo apt update
sudo apt install -y python3 python3-pip git curl
```

#### 2. ⚡ Install `uv` (Ultrafast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. 📂 Clone the Project Repository

Choose either method:

**HTTPS:**

```bash
git clone https://github.com/althaf-07/loan-sanction-prediction.git
```

**SSH:**

```bash
git clone git@github.com:althaf-07/loan-sanction-prediction.git
```

Then navigate into the project folder:

```bash
cd loan-sanction-prediction
```

#### 4. 📅 Install Project Dependencies

This will create a virtual environment and install all required packages:

```bash
uv sync
```

#### 5. 📊 Download Dataset

##### Option A: **GUI Method**

1. Open the following link in your browser and download the CSV file:
   - [Download entire_data.csv](https://drive.google.com/file/d/1L87oVoCRJ-JWTHoq1t79vJXwFBEVPvPI/view?usp=drive_link)
2. Move the file from your downloads folder to the project's raw data directory:

    ```bash
    mv ~/Downloads/entire_data.csv data/raw/
    ```

##### Option B: **Terminal Method (Automated)**

```bash
uv tool install gdown

# Download and move to data/raw
gdown --fuzzy "https://drive.google.com/file/d/1L87oVoCRJ-JWTHoq1t79vJXwFBEVPvPI/view?usp=drive_link"
mkdir -p data/raw/
mv entire_data.csv data/raw/
```

---

## ✨ Get Started

```bash
uv sync
uv run src/loan_sanction_prediction/split_data.py
uv run src/loan_sanction_prediction/preprocess_data.py
uv run src/loan_sanction_prediction/train.py
```

To run backend + UI:

```bash
nohup uvicorn src.loan_sanction_prediction.predict:app --reload > logs/uvicorn.log 2>&1 &
nohup streamlit run src/loan_sanction_prediction/streamlit_app.py > logs/streamlit.log 2>&1 &
```

---

## ♻️ Reproducibility Notes

- Set seed (`random_state=37`) wherever applicable to ensure consistent results.
- Use `uv sync` to guarantee the same dependency versions across machines.
- Configurations and paths are managed via [`config.yaml`](../src/loan_sanction_prediction/config.yaml)

---

## 🧹 Pre-commit Hooks

To maintain code quality and consistency, this project uses [pre-commit](https://pre-commit.com/). These hooks run automatically before each commit to catch common issues and enforce standards.

### ✨ Setup Instructions

```bash
uv add --dev pre-commit # First, ensure `pre-commit` is installed via `uv`
uv run pre-commit install # Install the git hooks
uv run pre-commit run --all-files # Run all hooks manually (recommended on first setup)
```

### 🚀 Active Hooks

Configured in the [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) file:

- `ruff` - Lint and fix code
- `ruff-format` - Format code
- `trailing-whitespace` - Remove trailing spaces
- `end-of-file-fixer` - Ensure every file ends with a newline
- `check-yaml` - Validate YAML file format
- `check-added-large-files` - Prevent committing large files

You can view and customize the hooks in the `.pre-commit-config.yaml` file.

### ✅ Usage Tips

- Hooks run automatically when you try to `git commit`
- To manually lint or format your code, you can run:

```bash
uv run pre-commit run --all-files
```

- If a hook fails, fix the issue and re-stage the changes before committing again
- For more hooks and usage tips, visit [pre-commit hooks](https://pre-commit.com/hooks.html).
- `pre-commit` hooks ensures code cleanliness and consistency across all contributors and environments.

---
