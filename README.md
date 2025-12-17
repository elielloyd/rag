# TrueClaim Vehicle Damage Analysis

A FastAPI application for vehicle damage analysis using Google Gemini API and AWS S3. The system classifies vehicle images by side, analyzes damage using AI, and stores results in Qdrant vector database for semantic search.

## Features

- **Image Classification**: Classify vehicle images by side (front, rear, left, right, roof)
- **Damage Analysis**: AI-powered damage description using Gemini API
- **S3 Integration**: Read vehicle images directly from AWS S3 buckets
- **Vector Storage**: Store damage descriptions in Qdrant with Gemini embeddings for semantic search
- **Structured Output**: Consistent JSON output matching insurance claim formats
- **REST API**: Full FastAPI application with OpenAPI documentation

## Project Structure

```
trueclaim-preprocessing/
├── config/                 # Configuration settings
│   ├── __init__.py
│   └── settings.py
├── models/                 # Pydantic models
│   ├── __init__.py
│   ├── api_models.py
│   └── vehicle_damage.py
├── prompts/                # Prompt templates
│   ├── __init__.py
│   └── vehicle_damage.py
├── routes/                 # API routes
│   ├── __init__.py
│   ├── health.py
│   ├── vehicle_damage.py
│   └── qdrant.py
├── services/               # Business logic
│   ├── __init__.py
│   ├── s3_service.py
│   ├── vehicle_damage_service.py
│   └── qdrant_service.py
├── .env.example           # Environment variables template
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── main.py                # FastAPI application
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to the project
cd trueclaim-preprocessing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` with your credentials:

```env
GEMINI_API_KEY=your_gemini_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
```

### 3. Start the Application

```bash
# Start Qdrant (for vector storage)
docker-compose up qdrant -d

# Run the FastAPI application
python main.py
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Vehicle Damage Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/vehicle-damage/classify` | Classify images by vehicle side |
| POST | `/vehicle-damage/analyze-side` | Analyze damage for a specific side |
| POST | `/vehicle-damage/analyze-classified` | Analyze pre-classified images |

### Qdrant Vector Database

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/qdrant/search` | Search for similar damage descriptions |
| GET | `/qdrant/collection/info` | Get collection information |
| DELETE | `/qdrant/collection` | Delete the collection |

## Usage Examples

### Step 1: Classify Images by Side

```bash
curl -X POST "http://localhost:8000/vehicle-damage/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": [
      "s3://bucket-name/claims/claim-123/image1.jpg",
      "s3://bucket-name/claims/claim-123/image2.jpg"
    ]
  }'
```

Response:
```json
{
  "success": true,
  "classified_images": {
    "front": ["s3://bucket-name/claims/claim-123/image1.jpg"],
    "rear": ["s3://bucket-name/claims/claim-123/image2.jpg"]
  }
}
```

### Step 2: Analyze Damage for a Side

```bash
curl -X POST "http://localhost:8000/vehicle-damage/analyze-side" \
  -H "Content-Type: application/json" \
  -d '{
    "side": "rear",
    "images": ["s3://bucket-name/claims/claim-123/rear1.jpg"],
    "vehicle_info": {
      "vin": "1HGBH41JXMN109186",
      "make": "Honda",
      "model": "Civic",
      "year": 2023,
      "body_type": "Sedan"
    },
    "approved_estimate": {
      "Bumper Cover Rear": [
        {"Description": "Bumper Cover Rear", "Operation": "Remove / Replace"},
        {"Description": "Bumper Cover Rear", "Operation": "Refinish"}
      ]
    }
  }'
```

### Search Similar Damage Descriptions

```bash
curl "http://localhost:8000/qdrant/search?query=rear%20bumper%20damage&limit=5"
```

## Output Format (ChunkOutput)

The damage analysis returns a structured `ChunkOutput`:

```json
{
  "vehicle_info": {
    "vin": "1HGBH41JXMN109186",
    "make": "Honda",
    "model": "Civic",
    "year": 2023,
    "body_type": "Sedan"
  },
  "side": "rear",
  "images": ["s3://bucket/image1.jpg", "s3://bucket/image2.jpg"],
  "damage_descriptions": [
    {
      "location": "Rear",
      "part": "Bumper Cover",
      "severity": "Moderate",
      "type": "Dent",
      "start_position": "Center",
      "end_position": "Right",
      "description": "Visible dent on rear bumper cover..."
    }
  ],
  "merged_damage_description": "The rear of the vehicle shows moderate damage...",
  "approved_estimate": {
    "Bumper Cover Rear": [
      {"Description": "Bumper Cover Rear", "Operation": "Remove / Replace"}
    ]
  }
}
```

## Qdrant Integration

Damage analysis results are automatically saved to Qdrant:
- **Content**: `merged_damage_description` is embedded using Gemini embeddings (768 dimensions)
- **Metadata**: Vehicle info, side, images, damage descriptions, and approved estimate

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key (required) |
| `AWS_ACCESS_KEY_ID` | - | AWS access key for S3 |
| `AWS_SECRET_ACCESS_KEY` | - | AWS secret key for S3 |
| `AWS_REGION` | `us-east-1` | AWS region |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION_NAME` | `image_descriptions` | Default collection name |
| `APP_HOST` | `0.0.0.0` | Application host |
| `APP_PORT` | `8000` | Application port |

## License

MIT
