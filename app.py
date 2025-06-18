import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
import json
import uuid
import time
import pickle
import os
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎬 AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .movie-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
    }
    
    .movie-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .movie-meta {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .movie-description {
        color: #34495e;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .rating-badge {
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: #2c3e50;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    .genre-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }
    
    .loading-container {
        text-align: center;
        padding: 3rem 0;
    }
    
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-ready {
        background-color: #28a745;
    }
    
    .status-pending {
        background-color: #ffc107;
    }
    
    .status-error {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class OptimizedMovieRecommenderRAG:
    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self.collection_name = "movies"
        self.movies_df = None
        self.embeddings = None
        
        # File paths for caching
        self.data_dir = Path("movie_data")
        self.data_dir.mkdir(exist_ok=True)
        self.embeddings_file = self.data_dir / "movie_embeddings.pkl"
        self.processed_data_file = self.data_dir / "processed_movies.pkl"
        
        # System status
        self.status = {
            'model_loaded': False,
            'database_connected': False,
            'data_processed': False,
            'embeddings_cached': False,
            'vector_db_populated': False,
            'ready_for_recommendations': False
        }
    
    def get_system_status(self) -> Dict[str, str]:
        """Get current system status with emojis"""
        status_map = {
            'model_loaded': '🤖 AI Model',
            'database_connected': '🗄️ Vector Database',
            'data_processed': '📊 Data Processing',
            'embeddings_cached': '💾 Embeddings Cache',
            'vector_db_populated': '🔄 Vector DB Population',
            'ready_for_recommendations': '🎯 Ready for Recommendations'
        }
        
        result = {}
        for key, label in status_map.items():
            status = self.status[key]
            icon = '✅' if status else '❌'
            result[label] = f"{icon} {'Ready' if status else 'Pending'}"
        
        return result
    
    def initialize_embedding_model(self):
        """Initialize a high-quality embedding model"""
        try:
            if self.embedding_model is None:
                with st.spinner("Loading high-quality AI embedding model..."):
                    # Using a better embedding model for improved accuracy
                    self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
                    # Alternative high-quality models:
                    # self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
                    # self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
                    
                self.status['model_loaded'] = True
                st.success("✅ High-quality embedding model loaded successfully!")
                logger.info("Embedding model initialized successfully")
            return True
        except Exception as e:
            st.error(f"❌ Error loading embedding model: {str(e)}")
            logger.error(f"Error initializing embedding model: {str(e)}")
            return False
    
    def initialize_vector_database(self):
        """Initialize Qdrant vector database connection"""
        try:
            if self.qdrant_client is None:
                qdrant_url = st.secrets["QDRANT_URL"]
                qdrant_api_key = st.secrets["QDRANT_API_KEY"]
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                )
                # Test connection
                collections = self.qdrant_client.get_collections()
                self.status['database_connected'] = True
                st.success("✅ Connected to vector database!")
                logger.info("Vector database connection established")
            return True
        except Exception as e:
            st.error(f"❌ Error connecting to vector database: {str(e)}")
            logger.error(f"Error connecting to vector database: {str(e)}")
            return False
    
    def load_and_process_data(self):
        """Load and process movie data with caching"""
        try:
            # Check if processed data exists
            if self.processed_data_file.exists():
                with st.spinner("Loading cached processed data..."):
                    with open(self.processed_data_file, 'rb') as f:
                        self.movies_df = pickle.load(f)
                    st.success(f"✅ Loaded {len(self.movies_df)} movies from cache!")
                    logger.info(f"Loaded processed data from cache: {len(self.movies_df)} movies")
            else:
                # Load and process fresh data
                with st.spinner("Processing movie data..."):
                    self.movies_df = pd.read_csv('movies.csv')
                    
                    # Enhanced text processing for better embeddings
                    self.movies_df['combined_text'] = (
                        "Movie: " + self.movies_df['movie_name'].fillna('') + ". " +
                        "Genre: " + self.movies_df['genre'].fillna('') + ". " +
                        "Plot: " + self.movies_df['description'].fillna('') + ". " +
                        "Director: " + self.movies_df['director'].fillna('') + ". " +
                        "Stars: " + self.movies_df['star'].fillna('') + ". " +
                        "Certificate: " + self.movies_df['certificate'].fillna('') + ". " +
                        "Rating: " + self.movies_df['rating'].astype(str)
                    )
                    
                    # Cache processed data
                    with open(self.processed_data_file, 'wb') as f:
                        pickle.dump(self.movies_df, f)
                    
                    st.success(f"✅ Processed and cached {len(self.movies_df)} movies!")
                    logger.info(f"Processed and cached {len(self.movies_df)} movies")
            
            self.status['data_processed'] = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            logger.error(f"Error processing data: {str(e)}")
            return False
    
    def create_or_load_embeddings(self):
        """Create embeddings or load from cache"""
        try:
            # Check if embeddings cache exists
            if self.embeddings_file.exists():
                with st.spinner("Loading embeddings from cache..."):
                    with open(self.embeddings_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.embeddings = cached_data['embeddings']
                        cached_movie_count = cached_data['movie_count']
                    
                    # Verify cache validity
                    if cached_movie_count == len(self.movies_df):
                        st.success(f"✅ Loaded {len(self.embeddings)} embeddings from cache!")
                        self.status['embeddings_cached'] = True
                        logger.info(f"Loaded embeddings from cache: {len(self.embeddings)} vectors")
                        return True
                    else:
                        st.warning("⚠️ Cache invalidated due to data changes. Regenerating embeddings...")
                        logger.warning("Embeddings cache invalidated due to data size mismatch")
            
            # Generate fresh embeddings
            with st.spinner("Generating embeddings for all movies..."):
                st.info("This may take a few minutes for the first run, but will be cached for future use.")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                texts = self.movies_df['combined_text'].tolist()
                batch_size = 32  # Optimal batch size for mpnet model
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts, 
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings)
                    
                    progress = min((i + batch_size) / len(texts), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} movies")
                
                self.embeddings = np.array(all_embeddings)
                
                # Cache embeddings
                cache_data = {
                    'embeddings': self.embeddings,
                    'movie_count': len(self.movies_df),
                    'embedding_model': 'all-mpnet-base-v2',
                    'created_at': time.time()
                }
                
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                st.success(f"✅ Generated and cached {len(self.embeddings)} embeddings!")
                logger.info(f"Generated and cached {len(self.embeddings)} embeddings")
            
            self.status['embeddings_cached'] = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating embeddings: {str(e)}")
            logger.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def setup_vector_database(self):
        """Setup Qdrant collection and populate if needed"""
        try:
            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),  # mpnet embeddings are 768-dim
                )
                st.success("✅ Created new vector collection!")
                logger.info("Created new vector collection")
            
            # Check if collection is populated
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            if collection_info.points_count == len(self.movies_df):
                st.info(f"ℹ️ Vector database already populated with {collection_info.points_count} movies")
                self.status['vector_db_populated'] = True
                logger.info(f"Vector database already populated: {collection_info.points_count} points")
                return True
            
            # Populate vector database
            with st.spinner("Populating vector database..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Clear existing data if count mismatch
                if collection_info.points_count > 0:
                    self.qdrant_client.delete_collection(self.collection_name)
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                    )
                
                batch_size = 100
                total_movies = len(self.movies_df)
                
                for i in range(0, total_movies, batch_size):
                    batch_df = self.movies_df.iloc[i:i+batch_size]
                    batch_embeddings = self.embeddings[i:i+batch_size]
                    
                    points = []
                    for idx, (_, row) in enumerate(batch_df.iterrows()):
                        point = PointStruct(
                            id=str(uuid.uuid4()),
                            vector=batch_embeddings[idx].tolist(),
                            payload={
                                "movie_name": str(row['movie_name']),
                                "certificate": str(row['certificate']),
                                "genre": str(row['genre']),
                                "rating": float(row['rating']) if pd.notna(row['rating']) else 0.0,
                                "description": str(row['description']),
                                "director": str(row['director']),
                                "star": str(row['star']),
                                "combined_text": str(row['combined_text']),
                                "original_index": int(row.name)
                            }
                        )
                        points.append(point)
                    
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    progress = min((i + batch_size) / total_movies, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Uploaded {min(i + batch_size, total_movies)}/{total_movies} movies")
                
                st.success("✅ Vector database populated successfully!")
                logger.info("Vector database populated successfully")
            
            self.status['vector_db_populated'] = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error setting up vector database: {str(e)}")
            logger.error(f"Error setting up vector database: {str(e)}")
            return False
    
    def initialize_complete_system(self):
        """Initialize the complete system with proper sequencing"""
        try:
            st.info("🚀 Starting complete system initialization...")
            
            # Step 1: Load embedding model
            if not self.initialize_embedding_model():
                return False
            
            # Step 2: Load and process data
            if not self.load_and_process_data():
                return False
            
            # Step 3: Create or load embeddings (this is the key optimization)
            if not self.create_or_load_embeddings():
                return False
            
            # Step 4: Initialize vector database
            if not self.initialize_vector_database():
                return False
            
            # Step 5: Setup vector database collection
            if not self.setup_vector_database():
                return False
            
            # Mark system as ready
            self.status['ready_for_recommendations'] = True
            st.success("🎉 System fully initialized and ready for recommendations!")
            logger.info("Complete system initialization successful")
            
            return True
            
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            logger.error(f"System initialization failed: {str(e)}")
            return False
    
    def get_recommendations(self, query: str, num_recommendations: int = 5) -> List[Dict]:
        """Get movie recommendations with LLM fallback for better quality"""
        try:
            if not self.status['ready_for_recommendations']:
                st.error("❌ System not ready. Please initialize the system first.")
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])
            
            # Search in Qdrant with many more results to account for aggressive filtering
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=num_recommendations * 15,  # Get even more results for filtering
                score_threshold=0.1  # Lower threshold to get more candidates
            )
            
            # Process and filter results with strict quality criteria
            recommendations = []
            seen_movies = set()
            
            # Sort by a combination of similarity score and rating for better results
            scored_results = []
            for result in search_results:
                movie_data = result.payload
                rating = float(movie_data.get('rating', 0))
                similarity = result.score
                
                # Combined score: 70% similarity + 30% normalized rating
                normalized_rating = rating / 10.0  # Normalize rating to 0-1 scale
                combined_score = (0.7 * similarity) + (0.3 * normalized_rating)
                
                scored_results.append((result, movie_data, combined_score))
            
            # Sort by combined score
            scored_results.sort(key=lambda x: x[2], reverse=True)
            
            # Apply strict quality filters
            for result, movie_data, combined_score in scored_results:
                movie_name = movie_data['movie_name']
                rating = float(movie_data.get('rating', 0))
                description = str(movie_data.get('description', '')).strip()
                director = str(movie_data.get('director', '')).strip()
                genre = str(movie_data.get('genre', '')).strip()
                
                # Very strict quality filters
                if (movie_name not in seen_movies and
                    rating >= 7.0 and  # Only movies rated 7.0+ from database
                    description and 
                    description not in ['Add a Plot', 'nan', 'N/A', '', 'No description available.'] and
                    len(description) > 40 and  # Substantial description required
                    director and 
                    director not in ['nan', 'N/A', '', 'Unknown', 'Not Available'] and
                    genre and 
                    genre not in ['nan', 'N/A', '', 'Unknown'] and
                    len(genre) > 3):  # Valid genre information
                    
                    movie_data['similarity_score'] = result.score
                    movie_data['combined_score'] = combined_score
                    movie_data['source'] = 'database'
                    recommendations.append(movie_data)
                    seen_movies.add(movie_name)
                    
                    if len(recommendations) >= num_recommendations:
                        break
            
            # If we don't have enough quality results from database, use LLM knowledge
            if len(recommendations) < max(2, num_recommendations // 2):  # Need at least 2 good movies or half the requested amount
                st.info(f"🤖 Database has limited high-quality matches for your query. Using AI knowledge base for better recommendations...")
                
                # Get LLM recommendations
                llm_recommendations = self.get_llm_movie_recommendations(query, num_recommendations - len(recommendations))
                
                # Add LLM recommendations
                for llm_movie in llm_recommendations:
                    if llm_movie['movie_name'] not in seen_movies:
                        llm_movie['source'] = 'llm_knowledge'
                        llm_movie['similarity_score'] = 0.95  # High confidence for LLM recommendations
                        recommendations.append(llm_movie)
                        seen_movies.add(llm_movie['movie_name'])
            
            # Final sort - prioritize database results but ensure good mix
            recommendations.sort(key=lambda x: (x.get('source') == 'database', x.get('combined_score', x.get('similarity_score', 0))), reverse=True)
            
            logger.info(f"Generated {len(recommendations)} recommendations for query: {query}")
            
            # Log the sources and quality for debugging
            if recommendations:
                db_count = len([r for r in recommendations if r.get('source') == 'database'])
                llm_count = len([r for r in recommendations if r.get('source') == 'llm_knowledge'])
                
                if db_count > 0:
                    db_ratings = [float(r.get('rating', 0)) for r in recommendations if r.get('source') == 'database']
                    avg_db_rating = sum(db_ratings) / len(db_ratings)
                    st.success(f"📊 Found {db_count} high-quality database movies (avg rating: {avg_db_rating:.1f}) + {llm_count} AI knowledge recommendations")
                else:
                    st.info(f"📊 All {llm_count} recommendations from AI knowledge base (database had no high-quality matches)")
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            st.error(f"❌ Error getting recommendations: {str(e)}")
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def get_llm_movie_recommendations(self, query: str, num_movies: int = 5) -> List[Dict]:
        """Get movie recommendations using LLM knowledge base"""
        try:
            api_url = "https://api.groq.com/openai/v1/chat/completions"
            groq_api_key = st.secrets["GROQ_API_KEY"]
            
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a movie expert. Recommend high-quality, well-known movies that match the user's request. 
                        
    CRITICAL: Respond ONLY with a JSON array of movie objects. Each movie must have:
    - movie_name: The exact movie title
    - rating: A realistic IMDb-style rating (6.5-9.5 range for good movies)
    - genre: Main genres separated by commas
    - description: 2-3 sentence plot summary (50-150 words)
    - director: The actual director's name
    - star: Main actors/actresses
    - certificate: Age rating (G, PG, PG-13, R, etc.)

    Focus on popular, critically acclaimed, or cult classic movies. No obscure or low-rated films."""
                    },
                    {
                        "role": "user",
                        "content": f"""Recommend {num_movies} high-quality movies for: "{query}"

    Return as JSON array only, no other text."""
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 0.9
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content'].strip()
                
                # Try to parse JSON response
                try:
                    # Clean up response in case it has markdown formatting
                    if '```json' in llm_response:
                        llm_response = llm_response.split('```json')[1].split('```')[0].strip()
                    elif '```' in llm_response:
                        llm_response = llm_response.split('```')[1].strip()
                    
                    movies_data = json.loads(llm_response)
                    
                    # Validate and format the data
                    formatted_movies = []
                    for movie in movies_data:
                        if isinstance(movie, dict) and 'movie_name' in movie:
                            formatted_movie = {
                                'movie_name': str(movie.get('movie_name', 'Unknown')),
                                'rating': float(movie.get('rating', 7.5)),
                                'genre': str(movie.get('genre', 'Drama')),
                                'description': str(movie.get('description', 'A great movie worth watching.')),
                                'director': str(movie.get('director', 'Various Directors')),
                                'star': str(movie.get('star', 'Talented Cast')),
                                'certificate': str(movie.get('certificate', 'PG-13')),
                                'combined_text': f"Movie: {movie.get('movie_name')}. Genre: {movie.get('genre')}. Plot: {movie.get('description')}. Director: {movie.get('director')}."
                            }
                            formatted_movies.append(formatted_movie)
                    
                    logger.info(f"LLM provided {len(formatted_movies)} movie recommendations")
                    return formatted_movies[:num_movies]
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM JSON response: {e}")
                    return self._get_fallback_recommendations(query, num_movies)
            
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return self._get_fallback_recommendations(query, num_movies)
                
        except Exception as e:
            logger.error(f"Error getting LLM recommendations: {str(e)}")
            return self._get_fallback_recommendations(query, num_movies)

    def _get_fallback_recommendations(self, query: str, num_movies: int = 5) -> List[Dict]:
        """Provide hardcoded fallback recommendations when LLM fails"""
        # Popular high-quality movies as fallback
        fallback_movies = [
            {
                'movie_name': 'The Shawshank Redemption',
                'rating': 9.3,
                'genre': 'Drama',
                'description': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                'director': 'Frank Darabont',
                'star': 'Tim Robbins, Morgan Freeman',
                'certificate': 'R',
                'combined_text': 'Movie: The Shawshank Redemption. Genre: Drama. Plot: Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.'
            },
            {
                'movie_name': 'The Godfather',
                'rating': 9.2,
                'genre': 'Crime, Drama',
                'description': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                'director': 'Francis Ford Coppola',
                'star': 'Marlon Brando, Al Pacino',
                'certificate': 'R',
                'combined_text': 'Movie: The Godfather. Genre: Crime, Drama. Plot: The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.'
            },
            {
                'movie_name': 'The Dark Knight',
                'rating': 9.0,
                'genre': 'Action, Crime, Drama',
                'description': 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.',
                'director': 'Christopher Nolan',
                'star': 'Christian Bale, Heath Ledger',
                'certificate': 'PG-13',
                'combined_text': 'Movie: The Dark Knight. Genre: Action, Crime, Drama. Plot: When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.'
            },
            {
                'movie_name': 'Pulp Fiction',
                'rating': 8.9,
                'genre': 'Crime, Drama',
                'description': 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
                'director': 'Quentin Tarantino',
                'star': 'John Travolta, Uma Thurman',
                'certificate': 'R',
                'combined_text': 'Movie: Pulp Fiction. Genre: Crime, Drama. Plot: The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.'
            },
            {
                'movie_name': 'Inception',
                'rating': 8.8,
                'genre': 'Action, Sci-Fi, Thriller',
                'description': 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into a CEO\'s mind.',
                'director': 'Christopher Nolan',
                'star': 'Leonardo DiCaprio, Marion Cotillard',
                'certificate': 'PG-13',
                'combined_text': 'Movie: Inception. Genre: Action, Sci-Fi, Thriller. Plot: A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into a CEO\'s mind.'
            }
        ]
        
        return fallback_movies[:num_movies]

    def generate_llm_explanation(self, query: str, recommendations: List[Dict]) -> str:
        """Generate explanation using Llama model via Groq API"""
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        groq_api_key = st.secrets["GROQ_API_KEY"]
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        if not recommendations:
            return "No quality movies found in our database that match your criteria with proper ratings and information."
        
        movies_context = "\n".join([
            f"- {movie['movie_name']} ({movie['genre']}) - Rating: {movie['rating']}/10 - {movie['description'][:150]}... - Director: {movie['director']}"
            for movie in recommendations[:3]  # Use top 3 movies
        ])
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a movie recommendation expert. These movies have already been filtered for quality (rating > 0, proper descriptions, valid directors). Provide a 2-3 sentence explanation of why these specific movies match the user's search, focusing on themes, genres, and story elements."""
                },
                {
                    "role": "user",
                    "content": f"""User searched for: "{query}"

    Quality movies from our database:
    {movies_context}

    Explain why these movies are good matches for the user's search."""
                }
            ],
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=15)
        result = response.json()
        explanation = result['choices'][0]['message']['content'].strip()
        
        return explanation

    
    def _generate_fallback_explanation(self, query: str, recommendations: List[Dict]) -> str:
        """Generate a fallback explanation when LLM API is not available"""
        if not recommendations:
            return f"No recommendations found for '{query}'. Try a different search term."
        
        genres = set()
        avg_rating = 0
        for movie in recommendations[:3]:
            if movie.get('genre'):
                genres.update([g.strip() for g in str(movie['genre']).split(',')])
            if movie.get('rating'):
                avg_rating += float(movie['rating'])
        
        avg_rating = avg_rating / min(len(recommendations), 3)
        genre_list = ", ".join(list(genres)[:3])
        
        return f"Based on your search for '{query}', I found movies that match your preferences. These recommendations include {genre_list} films with an average rating of {avg_rating:.1f}/10, selected for their thematic similarity and quality."

def display_movie_card(movie: Dict, rank: int):
    """Display a movie in a card format"""
    similarity_percentage = int(movie.get('similarity_score', 0) * 100)
    
    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">#{rank} {movie['movie_name']}</div>
        <div class="movie-meta">
            <span class="rating-badge">⭐ {movie['rating']}/10</span>
            <span style="margin-left: 10px;">📋 {movie.get('certificate', 'N/A')}</span>
            <span style="margin-left: 10px;">🎯 {similarity_percentage}% match</span>
        </div>
        <div style="margin: 10px 0;">
            {' '.join([f'<span class="genre-tag">{genre.strip()}</span>' for genre in str(movie.get('genre', '')).split(',') if genre.strip()])}
        </div>
        <div class="movie-description">{movie.get('description', 'No description available.')}</div>
        <div class="movie-meta">
            <strong>🎬 Director:</strong> {movie.get('director', 'N/A')}<br>
            <strong>⭐ Stars:</strong> {movie.get('star', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 AI-Powered Movie Recommender</h1>
        <p>Discover your next favorite movie with intelligent recommendations</p>
        <small>Optimized with caching for lightning-fast performance</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the recommender system
    if 'recommender' not in st.session_state:
        st.session_state.recommender = OptimizedMovieRecommenderRAG()
    
    recommender = st.session_state.recommender
    
    # Sidebar for setup and controls
    with st.sidebar:
        st.header("🛠️ System Setup")
        
        # System status display
        status_dict = recommender.get_system_status()
        st.markdown("**System Status:**")
        for component, status in status_dict.items():
            st.markdown(f"• {component}: {status}")
        
        st.markdown("---")
        
        # Initialize system button
        if st.button("🚀 Initialize Complete System", type="primary"):
            recommender.initialize_complete_system()
        
        # Individual component buttons for debugging
        with st.expander("🔧 Advanced Controls"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Load Model"):
                    recommender.initialize_embedding_model()
                if st.button("📊 Process Data"):
                    recommender.load_and_process_data()
                if st.button("💾 Create Embeddings"):
                    if recommender.embedding_model and recommender.movies_df is not None:
                        recommender.create_or_load_embeddings()
                    else:
                        st.warning("Load model and data first!")
            
            with col2:
                if st.button("🗄️ Connect DB"):
                    recommender.initialize_vector_database()
                if st.button("🔄 Setup Vector DB"):
                    if recommender.qdrant_client and recommender.embeddings is not None:
                        recommender.setup_vector_database()
                    else:
                        st.warning("Connect DB and create embeddings first!")
                if st.button("🗑️ Clear Cache"):
                    if recommender.embeddings_file.exists():
                        recommender.embeddings_file.unlink()
                        st.success("Cache cleared!")
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        num_recommendations = st.slider("Number of recommendations", 3, 15, 5)
        
        # Cache information
        if recommender.embeddings_file.exists():
            cache_size = recommender.embeddings_file.stat().st_size / (1024 * 1024)  # MB
            st.info(f"💾 Embeddings cache: {cache_size:.1f} MB")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Find Your Perfect Movie")
        
        # Search interface
        query = st.text_input(
            "What kind of movie are you looking for?",
            placeholder="E.g., 'action movies with superheroes', 'romantic comedy', 'sci-fi thriller'...",
            help="Describe the type of movie, genre, mood, or specific elements you're interested in"
        )
        
        col_search, col_examples = st.columns([1, 1])
        
        with col_search:
            search_button = st.button("🎯 Get Recommendations", type="primary")
        
        with col_examples:
            example_queries = [
                "action movies with great visual effects",
                "romantic comedies with happy endings", 
                "psychological thrillers that keep you guessing",
                "family-friendly animated movies",
                "sci-fi movies about space exploration",
                "crime dramas with complex characters",
                "movies like Inception with mind-bending plots",
                "feel-good movies for a bad day"
            ]
            
            selected_example = st.selectbox("Or try an example:", [""] + example_queries)
            if selected_example:
                query = selected_example
        
        # Get and display recommendations
        if (search_button and query.strip()) or (selected_example and query.strip()):
            if recommender.status['ready_for_recommendations']:
                with st.spinner("🔍 Finding perfect movies for you..."):
                    start_time = time.time()
                    recommendations = recommender.get_recommendations(query, num_recommendations)
                    search_time = time.time() - start_time
                    
                    if recommendations:
                        # Generate LLM explanation
                        explanation = recommender.generate_llm_explanation(query, recommendations)
                        
                        st.success(f"🎉 Found {len(recommendations)} great recommendations in {search_time:.2f}s!")
                        
                        # Display explanation
                        st.markdown("### 🤖 Why These Movies?")
                        st.info(explanation)
                        
                        # Display recommendations
                        st.markdown("### 🎬 Your Personalized Recommendations")
                        
                        for idx, movie in enumerate(recommendations, 1):
                            display_movie_card(movie, idx)
                            
                    else:
                        st.warning("🤔 No recommendations found. Try a different search term!")
            else:
                st.warning("⚠️ Please initialize the system first using the sidebar!")
    
    with col2:
        st.header("💡 Tips & Tricks")
        st.markdown("""
        **🎯 Better Search Tips:**
        - Be specific about genres or themes
        - Mention mood or atmosphere
        - Include director/actor names
        - Describe the experience you want
        - Use movie comparisons ("like X movie")
        
        **🔥 Popular Searches:**
        - "Marvel superhero movies"
        - "Movies like Inception"
        - "Feel-good comedies"
        - "Award-winning dramas"
        - "Movies with plot twists"
        
        **⚡ Performance Features:**
        - Embeddings are cached for speed
        - High-quality AI model (mpnet-base-v2)
        - Optimized vector search
        - Smart duplicate filtering
        """)
        
        if recommender.movies_df is not None:
            st.header("📈 Dataset Insights")
            
            # Show some basic stats
            avg_rating = recommender.movies_df['rating'].mean()
            top_genres = recommender.movies_df['genre'].str.split(',').explode().str.strip().value_counts().head(3)
            
            st.metric("Average Rating", f"{avg_rating:.1f}/10")
            st.metric("Total Movies", len(recommender.movies_df))
            
            st.markdown("**Top Genres:**")
            for genre, count in top_genres.items():
                st.markdown(f"• {genre}: {count} movies")

if __name__ == "__main__":
    main()