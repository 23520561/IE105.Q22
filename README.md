# IE105.Q22

# Backend Setup

## 1. Clone Repository

```bash
git clone https://github.com/23520561/IE105.Q22.git
cd IE105.Q22
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Mac/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

---

## 3. Install Poetry

```bash
pip install poetry
```

---

## 4. Install Dependencies

```bash
poetry install
```

---

## 5. Run the Project

```bash
poetry run uvicorn main:app --reload
```

---

## 6. Run Tests

```bash
poetry run pytest
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/23520561/IE105.Q22.git
cd IE105.Q22
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install poetry
poetry install
poetry run uvicorn main:app --reload
```

