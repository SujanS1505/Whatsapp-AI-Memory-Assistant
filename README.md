# WhatsApp Group AI Memory Assistant

An AI-powered system that listens to WhatsApp group messages, stores them in MongoDB, generates OpenAI embeddings, and allows group members to query the discussion history using natural language — directly inside the group chat.

---

## Architecture

```
WhatsApp Group
      │
      ▼
[Node.js Bot]  ──────────── POST /messages ──────────►  [FastAPI Backend]
 whatsapp-web.js                                           │
      ▲                                                    ├─ MongoDB  (message storage)
      │                                                    ├─ ChromaDB (vector search)
      └─────────────── AI Answer ◄── POST /query ──────── └─ OpenAI   (embeddings + LLM)
```

---

## Project Structure

```
whatsapp-ai-assistant/
├── whatsapp-bot/               # Node.js — WhatsApp integration
│   ├── src/
│   │   ├── whatsappClient.js   # WhatsApp client, QR login, event loop
│   │   ├── messageHandler.js   # Message routing & query detection
│   │   ├── apiClient.js        # HTTP client for FastAPI backend
│   │   └── config.js           # Environment-based configuration
│   ├── package.json
│   └── .env.example
│
└── ai-backend/                 # Python — AI pipeline
    ├── app/
    │   ├── main.py             # FastAPI app, lifespan, routers
    │   ├── config.py           # Pydantic settings
    │   ├── routes/
    │   │   ├── message_routes.py
    │   │   └── query_routes.py
    │   ├── services/
    │   │   ├── message_processor.py   # 3-step ingestion pipeline
    │   │   ├── embedding_service.py   # OpenAI embeddings
    │   │   ├── vector_service.py      # ChromaDB store & search
    │   │   ├── query_service.py       # RAG orchestrator
    │   │   └── summarization_service.py
    │   ├── database/
    │   │   ├── mongo_client.py
    │   │   └── message_repository.py
    │   ├── models/
    │   │   └── message_model.py
    │   └── utils/
    │       └── logger.py
    ├── requirements.txt
    └── .env.example
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Node.js | >= 18 |
| Python | >= 3.11 |
| MongoDB | >= 6.0 (local or Atlas) |
| OpenAI API Key | — |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/whatsapp-ai-assistant.git
cd whatsapp-ai-assistant
```

### 2. Set up the AI Backend (Python)

```bash
cd ai-backend

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set up the WhatsApp Bot (Node.js)

```bash
cd ../whatsapp-bot
npm install
```

---

## Environment Setup

### AI Backend

```bash
cd ai-backend
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
```

Edit `.env` and fill in all required values:

```env
OPENAI_API_KEY=sk-...
MONGODB_URI=mongodb://localhost:27017
CHROMA_DB_PATH=./chroma_data
```

### WhatsApp Bot

```bash
cd whatsapp-bot
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
```

Edit `.env`:

```env
WHATSAPP_SESSION=whatsapp-ai-session
BACKEND_API_URL=http://localhost:8000
```

---

## Running the System

### Step 1 — Start MongoDB

Ensure MongoDB is running locally, or provide an Atlas connection URI in `.env`.

```bash
# Run MongoDB directly (recommended — works without admin rights)
mongod --dbpath "C:\data\db"

# Alternative: start as Windows service (requires Administrator terminal)
net start MongoDB

# Alternative: Docker
docker run -d -p 27017:27017 --name mongo mongo:7
```

### Step 2 — Start the AI Backend

```bash
cd ai-backend

# Activate venv if not already active
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS / Linux

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend API will be available at: `http://localhost:8000`

Interactive API docs: `http://localhost:8000/docs`

### Step 3 — Start the WhatsApp Bot

Open a separate terminal:

```bash
cd whatsapp-bot
npm start
```

### Step 4 — Connect WhatsApp via QR Code

When the bot starts, a QR code will be printed in the terminal.

1. Open WhatsApp on your phone.
2. Go to **Linked Devices → Link a Device**.
3. Scan the QR code in the terminal.
4. The terminal will print: `[WhatsApp] Client is ready and connected!`

Your session is now saved locally. You will not need to scan again unless the session expires.

---

## Using the Assistant in a Group

Send messages in any WhatsApp group that the linked account is a member of.

| Command | Description |
|---|---|
| `@assistant summarize today's discussion` | Daily summary of the last 24 hours |
| `@assistant daily summary` | Same as above |
| `@assistant what did Chin say about commits` | Semantic search + RAG answer |
| `@assistant what tasks were assigned today` | Retrieves task-related messages |
| `@assistant what was discussed about the repository` | Topic Q&A |
| `@assistant search commits` | Returns top matching messages about commits |
| `@assistant summarize commits` | LLM summary of all commit-related messages |

The assistant **only responds when a message starts with `@assistant`**. All other messages are silently stored and indexed.

---

## API Reference

### POST `/messages`

Ingest a new group message.

**Request body:**
```json
{
  "group_id": "120363000000000000@g.us",
  "sender": "Chin",
  "message": "I just pushed the hotfix to the main branch",
  "timestamp": "2026-03-12T10:30:00Z"
}
```

**Response:**
```json
{
  "status": "ok",
  "message_id": "65f1a2b3c4d5e6f7a8b9c0d1",
  "detail": "Message stored and embedded successfully."
}
```

---

### POST `/query`

Query the assistant about past discussions.

**Request body:**
```json
{
  "group_id": "120363000000000000@g.us",
  "question": "what did Chin say about commits"
}
```

**Response:**
```json
{
  "answer": "Chin mentioned that he pushed a hotfix to the main branch...",
  "sources_count": 7
}
```

---

## Testing

### Manual API test with curl

```bash
# Ingest a test message
curl -X POST http://localhost:8000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "test-group-001",
    "sender": "Alice",
    "message": "We need to fix the authentication bug before Friday",
    "timestamp": "2026-03-12T09:00:00Z"
  }'

# Query the assistant
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "test-group-001",
    "question": "what bugs were mentioned"
  }'
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## Troubleshooting

**WhatsApp QR code not appearing**
- Ensure Puppeteer's Chromium dependencies are installed.
- On Windows, no additional steps are needed. On Linux, install Chromium dependencies:
  ```bash
  apt-get install -y libgbm-dev libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libpango-1.0-0 libcairo2 libasound2
  ```

**MongoDB connection refused**
- Verify MongoDB is running: `mongosh --eval "db.adminCommand('ping')"`

**OpenAI API errors**
- Confirm your API key is valid and has sufficient credits.
- Check rate limits if sending many messages at once.

**ChromaDB errors on Windows**
- Ensure the `chroma_data` directory path has write permissions.
- Use forward slashes or raw strings in the path if needed.

---

## License

MIT
