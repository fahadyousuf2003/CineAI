# 🎬 AI Movie Recommender System

A sophisticated RAG (Retrieval-Augmented Generation) based movie recommendation system that uses vector embeddings and AI to provide personalized movie suggestions.

## ✨ Features

- **Intelligent Search**: Natural language queries for movie discovery
- **Vector Embeddings**: Uses sentence transformers for semantic understanding
- **Cloud Vector Storage**: Qdrant Cloud for scalable vector database
- **AI Explanations**: LLM-powered explanations for recommendations
- **Beautiful UI**: Modern Streamlit interface with responsive design
- **Free Resources**: Built entirely with free-tier services

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom CSS
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Database**: Qdrant Cloud (Free Tier)
- **LLM**: Hugging Face Inference API (Free Tier)
- **Data Processing**: Pandas, NumPy

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd movie-recommender
pip install -r requirements.txt
```

### 2. Create Directory Structure

```
movie-recommender/
├── app.py
├── movies.csv
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml
```

### 3. Get API Keys

#### Qdrant Cloud (Vector Database)
1. Go to [Qdrant Cloud](https://cloud.qdrant.io/)
2. Sign up for free account
3. Create a new cluster (free tier: 1GB storage)
4. Get your cluster URL and API key

#### Hugging Face (LLM API)
1. Go to [Hugging Face](https://huggingface.co/)
2. Create free account
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permission

### 4. Configure Secrets

Create `.streamlit/secrets.toml`:

```toml
QDRANT_URL = "https://your-cluster-url.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"
HUGGINGFACE_API_KEY = "your-huggingface-token"
```

### 5. Prepare Your Dataset

Ensure your `movies.csv` has these columns:
- `movie_name`: Title of the movie
- `certificate`: Rating/certification (G, PG, R, etc.)
- `genre`: Movie genres (comma-separated)
- `rating`: IMDb rating (0-10)
- `description`: Movie plot/description
- `director`: Director name(s)
- `star`: Main actors/stars

### 6. Run the Application

```bash
streamlit run app.py
```

## 📊 How It Works

### 1. Data Processing
- Loads movie dataset from CSV
- Combines relevant fields into searchable text
- Handles missing data gracefully

### 2. Embedding Generation
- Uses SentenceTransformers to create vector embeddings
- Processes movies in batches for efficiency
- Stores embeddings in Qdrant Cloud

### 3. Semantic Search
- Converts user queries to embeddings
- Performs cosine similarity search in vector space
- Returns most relevant movies with similarity scores

### 4. AI Enhancement
- Uses Hugging Face API for recommendation explanations
- Provides context-aware reasoning
- Fallback explanations for reliability

## 🎯 Usage Examples

**Genre-based searches:**
- "action movies with superheroes"
- "romantic comedies from the 90s"
- "psychological thrillers"

**Mood-based searches:**
- "feel-good family movies"
- "mind-bending sci-fi films"
- "heartwarming dramas"

**Specific queries:**
- "movies like Inception"
- "films directed by Christopher Nolan"
- "Tom Hanks comedy movies"

## 🔧 Configuration Options

### Embedding Model
- Default: `all-MiniLM-L6-v2` (384 dimensions, fast)
- Alternative: `all-mpnet-base-v2` (768 dimensions, more accurate)

### Vector Database Settings
- Collection name: configurable
- Distance metric: Cosine similarity
- Batch size: adjustable for performance

### UI Customization
- Number of recommendations: 3-10
- Custom CSS themes
- Responsive design elements

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secrets in dashboard
4. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## 🔍 Troubleshooting

### Common Issues

**Embedding Model Loading**
- Ensure sufficient memory (>2GB)
- Check internet connection for model download
- Use lighter models for resource constraints

**Qdrant Connection**
- Verify URL format (include https://)
- Check API key permissions
- Ensure cluster is active

**Search Not Working**
- Initialize system first using sidebar
- Check if embeddings are stored
- Verify collection exists

### Performance Optimization

**For Large Datasets**
- Increase batch size for embedding generation
- Use more powerful embedding models
- Consider data preprocessing/filtering

**For Better Accuracy**
- Use higher-dimensional embeddings
- Implement query expansion
- Add metadata filtering

## 📈 Scaling Up

### Enhanced Features
- User preference learning
- Collaborative filtering
- Real-time recommendations
- Advanced filtering options

### Production Considerations
- Database connection pooling
- Caching strategies
- Load balancing
- Monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Add tests
5. Submit pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Open a GitHub issue
- Review Streamlit/Qdrant documentation

---

**Enjoy discovering your next favorite movie! 🍿**