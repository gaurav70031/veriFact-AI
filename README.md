# Fake News Detection

A full-stack machine learning application that detects fake news articles using NLP techniques.

## Architecture

```
fake-news-detection/
├── frontend/       # React + Vite UI
├── backend/        # FastAPI REST API
├── ml/             # ML model training & inference
├── database/       # SQL schema & migrations
├── scripts/        # Setup and utility scripts
├── docs/           # Project documentation
├── tests/          # Integration & e2e tests
└── docker/         # Docker configs
```

## Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Frontend  | React 18, Vite, Tailwind CSS        |
| Backend   | FastAPI, Python 3.11, SQLAlchemy    |
| ML        | scikit-learn, NLTK, Transformers    |
| Database  | PostgreSQL 15                       |
| Cache     | Redis 7                             |
| Container | Docker, Docker Compose              |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local ML dev)
- Node.js 18+ (for local frontend dev)

### 1. Clone and configure
```bash
git clone <repo-url>
cd fake-news-detection
cp .env.example .env
# Edit .env with your credentials
```

### 2. Train the model (first time)
```bash
cd ml
pip install -r requirements.txt
python train.py
```

### 3. Run with Docker
```bash
docker-compose up --build
```

The app will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 4. Run locally (development)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Method | Endpoint                  | Description                    |
|--------|---------------------------|--------------------------------|
| POST   | `/api/v1/predict`         | Predict if an article is fake  |
| GET    | `/api/v1/history`         | Get prediction history         |
| GET    | `/api/v1/stats`           | Model accuracy & usage stats   |
| GET    | `/api/v1/health`          | Health check                   |

## Model

The ML pipeline uses:
1. Text preprocessing (tokenization, stopword removal, lemmatization)
2. TF-IDF vectorization
3. Logistic Regression / PassiveAggressiveClassifier ensemble
4. Optional: fine-tuned BERT for higher accuracy

Training dataset: [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php)

## Testing

```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm run test

# ML tests
cd ml && pytest
```

## License

MIT
