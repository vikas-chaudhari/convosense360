import streamlit as st
import sentiment_helper
import sentiment_preprocessor
import pandas as pd
from googletrans import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
import time
import logging
from functools import lru_cache
import hashlib
import re
import emoji
import numpy as np
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import scipy.stats as stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Advanced Configuration
class SentimentConfig:
    CHUNK_SIZE = 500
    MAX_RETRIES = 5
    THREAD_POOL_SIZE = 8
    CACHE_SIZE = 5000
    MIN_CONFIDENCE_THRESHOLD = 0.1
    TEMPORAL_WINDOW_HOURS = 24
    SOCIAL_INFLUENCE_DECAY = 0.85
    EMOTION_MOMENTUM_FACTOR = 0.3

@dataclass
class EmotionProfile:
    """Advanced emotion profiling for users"""
    user_id: str
    dominant_emotion: str
    emotion_variance: float
    social_influence_score: float
    temporal_consistency: float
    communication_style: str
    emotional_triggers: List[str]
    sentiment_trajectory: List[float]

@dataclass
class ConversationContext:
    """Context-aware conversation analysis"""
    thread_id: str
    participants: List[str]
    topic_keywords: List[str]
    emotional_flow: List[float]
    turning_points: List[int]
    social_dynamics: Dict[str, float]

# Enhanced Emoji Sentiment with Context Awareness
ADVANCED_EMOJI_SENTIMENTS = {
    # Ultra Positive (Celebration, Love, Achievement)
    '🎉': {'base': 0.9, 'context_multiplier': {'celebration': 1.2, 'achievement': 1.1}},
    '❤️': {'base': 0.85, 'context_multiplier': {'love': 1.3, 'appreciation': 1.1}},
    '😍': {'base': 0.9, 'context_multiplier': {'admiration': 1.2, 'attraction': 1.1}},
    '💯': {'base': 0.8, 'context_multiplier': {'approval': 1.2, 'perfection': 1.3}},
    '🔥': {'base': 0.75, 'context_multiplier': {'excitement': 1.2, 'trend': 1.0}},
    '🚀': {'base': 0.8, 'context_multiplier': {'progress': 1.2, 'success': 1.1}},
    
    # Positive (Joy, Contentment)
    '😊': {'base': 0.6, 'context_multiplier': {'gratitude': 1.1, 'politeness': 0.9}},
    '😂': {'base': 0.8, 'context_multiplier': {'humor': 1.2, 'sarcasm': 0.7}},
    '👍': {'base': 0.5, 'context_multiplier': {'approval': 1.1, 'acknowledgment': 0.9}},
    '🙌': {'base': 0.7, 'context_multiplier': {'celebration': 1.2, 'gratitude': 1.1}},
    
    # Ultra Negative (Anger, Disgust, Devastation)
    '😡': {'base': -0.9, 'context_multiplier': {'anger': 1.3, 'frustration': 1.1}},
    '💔': {'base': -0.85, 'context_multiplier': {'heartbreak': 1.3, 'disappointment': 1.1}},
    '😭': {'base': -0.8, 'context_multiplier': {'grief': 1.2, 'overwhelming': 1.1}},
    '🤮': {'base': -0.9, 'context_multiplier': {'disgust': 1.3, 'rejection': 1.2}},
    '💀': {'base': -0.7, 'context_multiplier': {'death': 1.2, 'humor': -0.3}},
    
    # Negative (Sadness, Disappointment)
    '😞': {'base': -0.6, 'context_multiplier': {'disappointment': 1.2, 'sadness': 1.1}},
    '😢': {'base': -0.7, 'context_multiplier': {'sadness': 1.2, 'sympathy': 0.9}},
    '😠': {'base': -0.75, 'context_multiplier': {'anger': 1.2, 'annoyance': 1.0}},
    
    # Context-Dependent (Require sophisticated analysis)
    '🤔': {'base': 0.0, 'context_multiplier': {'confusion': -0.2, 'thinking': 0.1}},
    '😏': {'base': 0.0, 'context_multiplier': {'sarcasm': -0.3, 'confidence': 0.3}},
    '🙄': {'base': -0.4, 'context_multiplier': {'annoyance': 1.2, 'sarcasm': 1.0}},
}

# Advanced Linguistic Patterns with Psycholinguistic Analysis
PSYCHOLINGUISTIC_PATTERNS = {
    'cognitive_load': {
        'high': ['analyze', 'consider', 'evaluate', 'complex', 'intricate', 'sophisticated'],
        'low': ['simple', 'easy', 'obvious', 'clear', 'straightforward'],
        'score_impact': 0.2
    },
    'certainty_markers': {
        'high': ['definitely', 'absolutely', 'certainly', 'undoubtedly', 'clearly'],
        'low': ['maybe', 'perhaps', 'possibly', 'might', 'could be'],
        'score_impact': 0.3
    },
    'emotional_intensity': {
        'amplifiers': ['extremely', 'incredibly', 'absolutely', 'totally', 'completely'],
        'diminishers': ['somewhat', 'rather', 'quite', 'fairly', 'slightly'],
        'score_multiplier': 1.4
    },
    'social_orientation': {
        'inclusive': ['we', 'us', 'our', 'together', 'everyone'],
        'exclusive': ['I', 'me', 'my', 'myself', 'alone'],
        'score_impact': 0.25
    }
}

# Temporal Sentiment Patterns
TEMPORAL_PATTERNS = {
    'time_of_day': {
        'morning': {'energy_boost': 0.1, 'optimism_factor': 1.1},
        'afternoon': {'stability_factor': 1.0, 'neutrality_tendency': 0.05},
        'evening': {'reflection_boost': 0.1, 'emotional_depth': 1.1},
        'night': {'vulnerability_factor': 1.2, 'intensity_boost': 0.15}
    },
    'conversation_flow': {
        'opening': {'politeness_boost': 0.1},
        'middle': {'authenticity_factor': 1.1},
        'closing': {'gratitude_tendency': 0.1}
    }
}

class AdvancedSentimentEngine:
    """Revolutionary sentiment analysis engine with multi-dimensional analysis"""
    
    def __init__(self):
        self.cache = {}
        self.user_profiles = {}
        self.conversation_contexts = {}
        self.temporal_sentiment_buffer = deque(maxlen=1000)
        self.social_network = nx.DiGraph()
        self.topic_model = None
        self.sentiment_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def analyze_psycholinguistic_features(self, text: str) -> Dict[str, float]:
        """Advanced psycholinguistic analysis"""
        features = {}
        text_lower = text.lower()
        words = text_lower.split()
        
        for category, patterns in PSYCHOLINGUISTIC_PATTERNS.items():
            if category == 'emotional_intensity':
                amplifier_count = sum(1 for word in words if word in patterns['amplifiers'])
                diminisher_count = sum(1 for word in words if word in patterns['diminishers'])
                features[f'{category}_amplifiers'] = amplifier_count
                features[f'{category}_diminishers'] = diminisher_count
                features[f'{category}_net'] = amplifier_count - diminisher_count
            else:
                for subcategory, word_list in patterns.items():
                    if subcategory != 'score_impact' and subcategory != 'score_multiplier':
                        count = sum(1 for word in words if word in word_list)
                        features[f'{category}_{subcategory}'] = count
        
        return features
    
    def extract_contextual_emoji_sentiment(self, text: str, context: Dict = None) -> float:
        """Context-aware emoji sentiment analysis"""
        if not text:
            return 0
        
        total_score = 0
        emoji_count = 0
        
        for char in text:
            if char in ADVANCED_EMOJI_SENTIMENTS:
                emoji_data = ADVANCED_EMOJI_SENTIMENTS[char]
                base_score = emoji_data['base']
                
                # Apply context multipliers
                if context:
                    for ctx_key, multiplier in emoji_data.get('context_multiplier', {}).items():
                        if ctx_key in context.get('detected_contexts', []):
                            base_score *= multiplier
                            break
                
                total_score += base_score
                emoji_count += 1
        
        if emoji_count > 0:
            # Apply emoji clustering effect (multiple similar emojis amplify sentiment)
            clustering_factor = min(2.0, 1 + (emoji_count - 1) * 0.15)
            return (total_score / emoji_count) * clustering_factor
        
        return 0
    
    def detect_conversation_context(self, text: str) -> Dict[str, List[str]]:
        """Advanced context detection using multiple techniques"""
        contexts = {
            'detected_contexts': [],
            'topics': [],
            'emotions': [],
            'social_cues': []
        }
        
        text_lower = text.lower()
        
        # Topic detection patterns
        topic_patterns = {
            'work': ['job', 'office', 'boss', 'meeting', 'project', 'deadline', 'work'],
            'relationship': ['love', 'partner', 'boyfriend', 'girlfriend', 'date', 'marriage'],
            'family': ['mom', 'dad', 'sister', 'brother', 'family', 'parents', 'kids'],
            'health': ['doctor', 'hospital', 'sick', 'medicine', 'health', 'pain'],
            'celebration': ['birthday', 'party', 'celebrate', 'congratulations', 'wedding'],
            'achievement': ['success', 'won', 'passed', 'graduated', 'promotion', 'accomplished']
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                contexts['detected_contexts'].append(topic)
                contexts['topics'].append(topic)
        
        # Emotional context detection
        emotion_patterns = {
            'anger': ['angry', 'mad', 'furious', 'rage', 'annoyed'],
            'joy': ['happy', 'excited', 'thrilled', 'delighted', 'cheerful'],
            'sadness': ['sad', 'depressed', 'down', 'upset', 'crying'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished']
        }
        
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                contexts['emotions'].append(emotion)
        
        return contexts
    
    def calculate_social_influence_score(self, user: str, timestamp: datetime, 
                                       recent_messages: List[Dict]) -> float:
        """Calculate how much a user's sentiment is influenced by others"""
        if not recent_messages:
            return 0
        
        influence_score = 0
        user_messages = [msg for msg in recent_messages if msg['user'] == user]
        other_messages = [msg for msg in recent_messages if msg['user'] != user]
        
        if not user_messages or not other_messages:
            return 0
        
        # Analyze sentiment synchronization
        for user_msg in user_messages:
            user_time = user_msg['timestamp']
            
            # Find messages from others within temporal window
            recent_others = [
                msg for msg in other_messages 
                if abs((msg['timestamp'] - user_time).total_seconds()) < 3600  # 1 hour window
            ]
            
            if recent_others:
                # Calculate sentiment alignment
                user_sentiment = user_msg.get('sentiment_score', 0)
                other_sentiments = [msg.get('sentiment_score', 0) for msg in recent_others]
                avg_other_sentiment = np.mean(other_sentiments)
                
                # Measure alignment (inverse of difference)
                alignment = 1 - abs(user_sentiment - avg_other_sentiment) / 2
                influence_score += alignment
        
        return influence_score / len(user_messages) if user_messages else 0
    
    def analyze_temporal_sentiment_patterns(self, messages: List[Dict]) -> Dict:
        """Analyze sentiment patterns over time"""
        if not messages:
            return {}
        
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda x: x.get('timestamp', datetime.now()))
        
        patterns = {
            'sentiment_trajectory': [],
            'volatility_score': 0,
            'trend_direction': 'stable',
            'emotional_peaks': [],
            'recovery_patterns': []
        }
        
        sentiments = [msg.get('sentiment_score', 0) for msg in sorted_messages]
        patterns['sentiment_trajectory'] = sentiments
        
        if len(sentiments) > 1:
            # Calculate volatility
            patterns['volatility_score'] = np.std(sentiments)
            
            # Determine trend
            correlation = np.corrcoef(range(len(sentiments)), sentiments)[0, 1]
            if correlation > 0.1:
                patterns['trend_direction'] = 'improving'
            elif correlation < -0.1:
                patterns['trend_direction'] = 'declining'
            
            # Find emotional peaks and valleys
            for i in range(1, len(sentiments) - 1):
                if sentiments[i] > sentiments[i-1] and sentiments[i] > sentiments[i+1]:
                    if abs(sentiments[i]) > 0.5:  # Significant peak
                        patterns['emotional_peaks'].append({
                            'index': i,
                            'value': sentiments[i],
                            'type': 'positive' if sentiments[i] > 0 else 'negative'
                        })
        
        return patterns
    
    def nuclear_sentiment_analysis(self, text: str, user: str = None, 
                                 timestamp: datetime = None, context: Dict = None) -> Dict:
        """Ultimate sentiment analysis with all advanced features"""
        if not text or not isinstance(text, str):
            return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0, 'confidence': 0}
        
        # Initialize scores
        scores = {
            'emoji': 0,
            'psycholinguistic': 0,
            'contextual': 0,
            'temporal': 0,
            'social': 0,
            'linguistic_patterns': 0
        }
        
        # 1. Contextual emoji analysis
        detected_context = self.detect_conversation_context(text)
        scores['emoji'] = self.extract_contextual_emoji_sentiment(text, detected_context)
        
        # 2. Psycholinguistic features
        psych_features = self.analyze_psycholinguistic_features(text)
        
        # Convert psycholinguistic features to sentiment score
        certainty_high = psych_features.get('certainty_markers_high', 0)
        certainty_low = psych_features.get('certainty_markers_low', 0)
        intensity_net = psych_features.get('emotional_intensity_net', 0)
        
        scores['psycholinguistic'] = (
            (certainty_high - certainty_low) * 0.1 +
            intensity_net * 0.2
        )
        
        # 3. Advanced pattern matching
        text_lower = text.lower()
        
        # Sentiment amplification patterns
        amplification_patterns = {
            r'\b(so+|very|extremely|incredibly|absolutely)\s+(\w+)': 1.3,
            r'\b(totally|completely|utterly)\s+(\w+)': 1.2,
            r'\b(kind\s+of|sort\s+of|somewhat)\s+(\w+)': 0.7,
        }
        
        pattern_multiplier = 1.0
        for pattern, multiplier in amplification_patterns.items():
            if re.search(pattern, text_lower):
                pattern_multiplier *= multiplier
        
        # 4. Traditional sentiment analysis (VADER + TextBlob)
        try:
            vader_analyzer = get_sentiment_analyzer()
            vader_scores = vader_analyzer.polarity_scores(text)
            vader_compound = vader_scores['compound']
        except:
            vader_compound = 0
        
        try:
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
        except:
            textblob_score = 0
        
        # 5. Combine all scores with sophisticated weighting
        final_compound = (
            scores['emoji'] * 0.35 +                    # Emoji analysis (highest weight)
            scores['psycholinguistic'] * 0.20 +         # Psycholinguistic features
            vader_compound * 0.25 +                     # VADER (reliable baseline)
            textblob_score * 0.15 +                     # TextBlob
            scores['contextual'] * 0.05                 # Contextual adjustments
        ) * pattern_multiplier
        
        # Apply anti-neutrality system with more sophistication
        if abs(final_compound) < 0.05 and len(text.strip()) > 10:
            # Detect subtle sentiment indicators
            subtle_positive = ['thanks', 'good', 'nice', 'cool', 'ok', 'fine']
            subtle_negative = ['no', 'not', 'bad', 'wrong', 'problem', 'issue']
            
            if any(word in text_lower for word in subtle_positive):
                final_compound = max(final_compound, 0.2)
            elif any(word in text_lower for word in subtle_negative):
                final_compound = min(final_compound, -0.2)
        
        # Convert to pos/neg/neu scores
        if final_compound > 0.05:
            pos_score = min(abs(final_compound) * 1.5, 1.0)
            neg_score = 0
            neu_score = max(0, 1 - pos_score)
        elif final_compound < -0.05:
            neg_score = min(abs(final_compound) * 1.5, 1.0)
            pos_score = 0
            neu_score = max(0, 1 - neg_score)
        else:
            pos_score = 0
            neg_score = 0
            neu_score = 1
        
        # Calculate confidence score
        score_magnitude = abs(final_compound)
        emoji_boost = min(0.3, abs(scores['emoji']) * 0.5)
        confidence = min(1.0, score_magnitude + emoji_boost)
        
        return {
            'pos': pos_score,
            'neg': neg_score,
            'neu': neu_score,
            'compound': final_compound,
            'confidence': confidence,
            'components': scores,
            'context': detected_context,
            'psycholinguistic_features': psych_features
        }

# Initialize the advanced engine
@st.cache_resource
def get_advanced_sentiment_engine():
    return AdvancedSentimentEngine()

@st.cache_resource
def get_translator():
    return Translator()

@st.cache_resource  
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def init_session_state():
    """Initialize session state with advanced features"""
    defaults = {
        "show_sentiment": True,
        "processed_data": None,
        "file_hash": None,
        "user_profiles": {},
        "conversation_analysis": None,
        "topic_analysis": None,
        "temporal_analysis": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def advanced_parallel_processing(messages: List[str], users: List[str], 
                                timestamps: List[datetime] = None) -> List[Dict]:
    """Advanced parallel processing with sophisticated analysis"""
    if not messages:
        return []
    
    engine = get_advanced_sentiment_engine()
    results = []
    
    # Create batches for parallel processing
    batch_size = SentimentConfig.CHUNK_SIZE
    num_batches = (len(messages) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    def process_batch(batch_data):
        batch_results = []
        for i, (msg, user) in enumerate(batch_data):
            timestamp = timestamps[i] if timestamps else None
            
            try:
                # Translate if necessary
                translated_msg = safe_translate(msg)
                
                # Advanced sentiment analysis
                analysis = engine.nuclear_sentiment_analysis(
                    translated_msg, user, timestamp
                )
                
                batch_results.append({
                    'original_message': msg,
                    'translated_message': translated_msg,
                    'user': user,
                    'timestamp': timestamp,
                    'pos': analysis['pos'],
                    'neg': analysis['neg'],
                    'neu': analysis['neu'],
                    'compound': analysis['compound'],
                    'confidence': analysis['confidence'],
                    'components': analysis['components'],
                    'context': analysis['context'],
                    'psycholinguistic_features': analysis['psycholinguistic_features']
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Fallback
                batch_results.append({
                    'original_message': msg,
                    'translated_message': msg,
                    'user': user,
                    'timestamp': timestamp,
                    'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0,
                    'confidence': 0, 'components': {}, 'context': {}, 
                    'psycholinguistic_features': {}
                })
        
        return batch_results
    
    # Prepare batch data
    batch_data_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(messages))
        batch = list(zip(messages[start_idx:end_idx], users[start_idx:end_idx]))
        batch_data_list.append(batch)
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=SentimentConfig.THREAD_POOL_SIZE) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batch_data_list]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                
                progress = (i + 1) / len(futures)
                progress_bar.progress(progress)
                progress_text.text(f"🚀 Processing: {i + 1}/{len(futures)} batches completed")
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    progress_bar.empty()
    progress_text.empty()
    
    return results

@lru_cache(maxsize=SentimentConfig.CACHE_SIZE)
def cached_translate(text_hash: str, text: str) -> str:
    """Enhanced translation with better language detection"""
    if not text or len(text.strip()) < 3:
        return text
    
    # Enhanced English detection
    english_indicators = [
        r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',
        r'\b(is|are|was|were|have|has|had|will|would|can|could)\b',
        r'\b(this|that|these|those|what|when|where|why|how)\b'
    ]
    
    english_matches = sum(len(re.findall(pattern, text.lower())) for pattern in english_indicators)
    words = len(text.split())
    
    if words > 0 and english_matches / words > 0.3:
        return text
    
    try:
        translator = get_translator()
        result = translator.translate(text, dest='en')
        return result.text if result and hasattr(result, 'text') else text
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text

def safe_translate(msg: str) -> str:
    """Safe translation wrapper"""
    if not isinstance(msg, str) or not msg.strip():
        return msg
    
    msg_hash = hashlib.md5(msg.encode()).hexdigest()
    return cached_translate(msg_hash, msg)

def perform_advanced_user_analysis(data: pd.DataFrame) -> Dict:
    """Advanced user profiling and analysis"""
    if data.empty:
        return {}
    
    user_analysis = {}
    
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        
        if len(user_data) < 5:  # Skip users with too few messages
            continue
        
        # Basic sentiment statistics
        pos_scores = user_data['pos'].values
        neg_scores = user_data['neg'].values
        compounds = user_data['compound'].values
        
        # Advanced metrics
        sentiment_variance = np.var(compounds)
        sentiment_range = np.max(compounds) - np.min(compounds)
        emotional_stability = 1 - (sentiment_variance / (sentiment_range + 0.001))
        
        # Determine dominant emotion
        avg_pos = np.mean(pos_scores)
        avg_neg = np.mean(neg_scores)
        
        if avg_pos > avg_neg + 0.1:
            dominant_emotion = "Optimistic"
        elif avg_neg > avg_pos + 0.1:
            dominant_emotion = "Pessimistic"
        else:
            dominant_emotion = "Balanced"
        
        # Communication style analysis
        avg_message_length = user_data['original_message'].str.len().mean()
        caps_usage = user_data['original_message'].str.count(r'[A-Z]').mean()
        punctuation_usage = user_data['original_message'].str.count(r'[!?.]').mean()
        
        if avg_message_length > 100 and punctuation_usage > 2:
            comm_style = "Expressive"
        elif avg_message_length < 30 and caps_usage < 2:
            comm_style = "Concise"
        elif caps_usage > 5 or punctuation_usage > 5:
            comm_style = "Intense"
        else:
            comm_style = "Casual"
        
        # Temporal consistency
        if 'timestamp' in user_data.columns:
            daily_sentiments = user_data.groupby(user_data['timestamp'].dt.date)['compound'].mean()
            temporal_consistency = 1 - (daily_sentiments.std() / (abs(daily_sentiments.mean()) + 0.001))
        else:
            temporal_consistency = 0.5
        
        user_analysis[user] = {
            'message_count': len(user_data),
            'avg_sentiment': np.mean(compounds),
            'sentiment_variance': sentiment_variance,
            'emotional_stability': emotional_stability,
            'dominant_emotion': dominant_emotion,
            'communication_style': comm_style,
            'temporal_consistency': temporal_consistency,
            'avg_confidence': user_data['confidence'].mean(),
            'pos_ratio': len(user_data[user_data['compound'] > 0.1]) / len(user_data),
            'neg_ratio': len(user_data[user_data['compound'] < -0.1]) / len(user_data),
            'context_diversity': len([ctx for contexts in user_data['context'] for ctx in contexts.get('topics', [])]),
        }
    
    return user_analysis

def create_advanced_visualizations(data: pd.DataFrame, user_analysis: Dict):
    """Create sophisticated visualizations"""
    
    # 1. Sentiment Heatmap by User and Time
    if 'timestamp' in data.columns and not data.empty:
        st.subheader("🔥 Temporal Sentiment Heatmap")
        
        # Create hourly sentiment matrix
        data['hour'] = data['timestamp'].dt.hour
        data['date'] = data['timestamp'].dt.date
        
        pivot_data = data.pivot_table(
            values='compound', 
            index='user', 
            columns='hour', 
            aggfunc='mean'
        ).fillna(0)
        
        fig = px.imshow(
            pivot_data.values,
            x=[f"{h:02d}:00" for h in pivot_data.columns],  # Use actual column hours
            y=pivot_data.index,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            title="Sentiment Intensity by User and Hour of Day"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. User Personality Radar Chart
    st.subheader("🎯 User Personality Profiles")
    
    if user_analysis:
        # Create radar chart for user personalities
        users = list(user_analysis.keys())[:6]  # Limit to top 6 users for clarity
        
        metrics = ['emotional_stability', 'temporal_consistency', 'avg_confidence', 
                  'pos_ratio', 'context_diversity']
        
        fig = go.Figure()
        
        for user in users:
            values = []
            for metric in metrics:
                value = user_analysis[user].get(metric, 0)
                # Normalize values to 0-1 scale
                if metric == 'context_diversity':
                    value = min(1.0, value / 10)  # Normalize diversity
                values.append(value)
            
            # Close the radar chart
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=user,
                line_color=px.colors.qualitative.Set3[len(fig.data) % len(px.colors.qualitative.Set3)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="User Personality Radar Chart",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. Sentiment Distribution with Advanced Statistics
    st.subheader("📊 Advanced Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Compound score distribution
        fig = px.histogram(
            data, 
            x='compound', 
            nbins=50,
            title="Sentiment Score Distribution",
            color_discrete_sequence=['#FF6B6B']
        )
        fig.add_vline(x=data['compound'].mean(), line_dash="dash", 
                     annotation_text=f"Mean: {data['compound'].mean():.3f}")
        fig.add_vline(x=data['compound'].median(), line_dash="dot", 
                     annotation_text=f"Median: {data['compound'].median():.3f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig = px.box(
            data, 
            y='confidence', 
            x='user' if len(data['user'].unique()) <= 10 else None,
            title="Confidence Score Distribution by User"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Advanced Time Series Analysis
    if 'timestamp' in data.columns:
        st.subheader("⏱️ Temporal Sentiment Evolution")
        
        # Resample data by hour for smoother visualization
        data_resampled = data.set_index('timestamp').resample('H')['compound'].agg(['mean', 'std', 'count'])
        data_resampled = data_resampled.dropna()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sentiment Over Time', 'Message Volume'),
            vertical_spacing=0.12
        )
        
        # Sentiment line with confidence bands
        fig.add_trace(
            go.Scatter(
                x=data_resampled.index,
                y=data_resampled['mean'] + data_resampled['std'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,80,0)',
                showlegend=False
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_resampled.index,
                y=data_resampled['mean'] - data_resampled['std'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0)',
                name='Confidence Band',
                fillcolor='rgba(0,100,80,0.2)'
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_resampled.index,
                y=data_resampled['mean'],
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color='#1f77b4', width=3)
            ), row=1, col=1
        )
        
        # Message volume
        fig.add_trace(
            go.Bar(
                x=data_resampled.index,
                y=data_resampled['count'],
                name='Message Count',
                marker_color='#ff7f0e',
                opacity=0.7
            ), row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Advanced Temporal Analysis")
        st.plotly_chart(fig, use_container_width=True)

def create_conversation_flow_analysis(data: pd.DataFrame):
    """Analyze conversation flow and dynamics"""
    if data.empty or 'timestamp' not in data.columns:
        return
    
    st.subheader("🌊 Conversation Flow Analysis")
    
    # Sort by timestamp
    data_sorted = data.sort_values('timestamp')
    
    # Calculate sentiment momentum (rate of change)
    data_sorted['sentiment_momentum'] = data_sorted['compound'].diff().fillna(0)
    
    # Detect conversation turning points
    turning_points = []
    window_size = 5
    
    for i in range(window_size, len(data_sorted) - window_size):
        before_window = data_sorted.iloc[i-window_size:i]['compound'].mean()
        after_window = data_sorted.iloc[i:i+window_size]['compound'].mean()
        
        if abs(after_window - before_window) > 0.3:  # Significant change
            turning_points.append({
                'index': i,
                'timestamp': data_sorted.iloc[i]['timestamp'],
                'change': after_window - before_window,
                'type': 'positive_shift' if after_window > before_window else 'negative_shift'
            })
    
    # Visualize conversation flow
    fig = go.Figure()
    
    # Main sentiment line
    fig.add_trace(go.Scatter(
        x=data_sorted['timestamp'],
        y=data_sorted['compound'],
        mode='lines+markers',
        name='Sentiment Flow',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4)
    ))
    
    # Add momentum as secondary trace
    fig.add_trace(go.Scatter(
        x=data_sorted['timestamp'],
        y=data_sorted['sentiment_momentum'],
        mode='lines',
        name='Sentiment Momentum',
        yaxis='y2',
        line=dict(color='#A23B72', width=1, dash='dash'),
        opacity=0.7
    ))
    
    # Mark turning points
    for tp in turning_points:
        fig.add_vline(
            x=tp['timestamp'],
            line_dash="dot",
            line_color="red" if tp['type'] == 'negative_shift' else "green",
            annotation_text=f"Turn: {tp['change']:.2f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Conversation Sentiment Flow with Turning Points",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        yaxis2=dict(
            title="Momentum",
            overlaying='y',
            side='right'
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display insights
    if turning_points:
        st.write("**Key Conversation Turning Points:**")
        for i, tp in enumerate(turning_points[:5]):  # Show top 5
            direction = "📈 Positive" if tp['type'] == 'positive_shift' else "📉 Negative"
            st.write(f"{i+1}. {direction} shift at {tp['timestamp'].strftime('%H:%M:%S')} (Δ{tp['change']:.2f})")

def create_topic_modeling_analysis(data: pd.DataFrame):
    """Advanced topic modeling and analysis"""
    if data.empty or len(data) < 10:
        return
    
    st.subheader("🎯 Topic Analysis & Sentiment Correlation")
    
    # Prepare text data
    texts = data['translated_message'].fillna('').tolist()
    texts = [text for text in texts if len(text.strip()) > 10]
    
    if len(texts) < 5:
        st.warning("Not enough text data for topic analysis")
        return
    
    try:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Topic modeling with LDA
        n_topics = min(5, len(texts) // 3)  # Dynamic topic count
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(tfidf_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weights': topic[top_words_idx]
            })
        
        # Display topics
        cols = st.columns(min(3, len(topics)))
        for i, topic in enumerate(topics):
            with cols[i % len(cols)]:
                st.write(f"**Topic {i+1}:**")
                for word, weight in zip(topic['words'][:5], topic['weights'][:5]):
                    st.write(f"• {word} ({weight:.3f})")
        
        # Topic-Sentiment correlation
        doc_topic_probs = lda.transform(tfidf_matrix)
        
        # Assign dominant topic to each message
        dominant_topics = np.argmax(doc_topic_probs, axis=1)
        
        # Calculate average sentiment per topic
        topic_sentiments = {}
        for topic_id in range(n_topics):
            topic_messages_idx = np.where(dominant_topics == topic_id)[0]
            if len(topic_messages_idx) > 0:
                topic_data = data.iloc[topic_messages_idx]
                topic_sentiments[topic_id] = {
                    'avg_sentiment': topic_data['compound'].mean(),
                    'message_count': len(topic_messages_idx),
                    'pos_ratio': (topic_data['compound'] > 0.1).mean(),
                    'neg_ratio': (topic_data['compound'] < -0.1).mean()
                }
        
        # Visualize topic-sentiment relationship
        if topic_sentiments:
            topic_ids = list(topic_sentiments.keys())
            avg_sentiments = [topic_sentiments[tid]['avg_sentiment'] for tid in topic_ids]
            message_counts = [topic_sentiments[tid]['message_count'] for tid in topic_ids]
            
            fig = go.Figure(data=go.Scatter(
                x=topic_ids,
                y=avg_sentiments,
                mode='markers+text',
                marker=dict(
                    size=[count*3 for count in message_counts],
                    color=avg_sentiments,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sentiment")
                ),
                text=[f"Topic {tid+1}" for tid in topic_ids],
                textposition="middle center"
            ))
            
            fig.update_layout(
                title="Topic Sentiment Analysis",
                xaxis_title="Topic ID",
                yaxis_title="Average Sentiment",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Topic modeling failed: {str(e)}")

def create_sentiment_insights_dashboard(data: pd.DataFrame, user_analysis: Dict):
    """Create comprehensive insights dashboard"""
    st.subheader("🧠 AI-Powered Insights Dashboard")
    
    if data.empty:
        return
    
    insights = []
    
    # Overall sentiment health
    avg_sentiment = data['compound'].mean()
    sentiment_std = data['compound'].std()
    
    if avg_sentiment > 0.2:
        insights.append(f"🌟 **Positive Community**: Overall sentiment is positive ({avg_sentiment:.3f})")
    elif avg_sentiment < -0.2:
        insights.append(f"⚠️ **Negative Trend**: Community sentiment is concerning ({avg_sentiment:.3f})")
    else:
        insights.append(f"⚖️ **Balanced Discussion**: Neutral sentiment with room for improvement ({avg_sentiment:.3f})")
    
    # Sentiment volatility
    if sentiment_std > 0.4:
        insights.append(f"🎢 **High Volatility**: Conversation has dramatic sentiment swings (σ={sentiment_std:.3f})")
    elif sentiment_std < 0.2:
        insights.append(f"🧘 **Stable Discussion**: Consistent emotional tone throughout (σ={sentiment_std:.3f})")
    
    # User participation analysis
    if user_analysis:
        most_positive_user = max(user_analysis.items(), key=lambda x: x[1]['avg_sentiment'])
        most_negative_user = min(user_analysis.items(), key=lambda x: x[1]['avg_sentiment'])
        most_active_user = max(user_analysis.items(), key=lambda x: x[1]['message_count'])
        
        insights.append(f"👑 **Most Positive**: {most_positive_user[0]} (avg: {most_positive_user[1]['avg_sentiment']:.3f})")
        insights.append(f"😔 **Most Critical**: {most_negative_user[0]} (avg: {most_negative_user[1]['avg_sentiment']:.3f})")
        insights.append(f"💬 **Most Active**: {most_active_user[0]} ({most_active_user[1]['message_count']} messages)")
        
        # Communication style distribution
        styles = [profile['communication_style'] for profile in user_analysis.values()]
        style_counts = Counter(styles)
        dominant_style = style_counts.most_common(1)[0]
        insights.append(f"🗣️ **Dominant Style**: {dominant_style[0]} communication ({dominant_style[1]} users)")
    
    # Confidence analysis
    high_confidence_ratio = (data['confidence'] > 0.7).mean()
    if high_confidence_ratio > 0.6:
        insights.append(f"✅ **High Confidence**: {high_confidence_ratio:.1%} of analyses are highly confident")
    else:
        insights.append(f"🤔 **Mixed Signals**: Only {high_confidence_ratio:.1%} of analyses are highly confident")
    
    # Time-based insights
    if 'timestamp' in data.columns:
        hourly_sentiment = data.groupby(data['timestamp'].dt.hour)['compound'].mean()
        best_hour = hourly_sentiment.idxmax()
        worst_hour = hourly_sentiment.idxmin()
        
        insights.append(f"🌅 **Peak Positivity**: {best_hour:02d}:00 (sentiment: {hourly_sentiment[best_hour]:.3f})")
        insights.append(f"🌙 **Lowest Point**: {worst_hour:02d}:00 (sentiment: {hourly_sentiment[worst_hour]:.3f})")
    
    # Display insights in columns
    col1, col2 = st.columns(2)
    
    for i, insight in enumerate(insights):
        if i % 2 == 0:
            col1.markdown(insight)
        else:
            col2.markdown(insight)

def main():
    """Enhanced main application"""
    st.set_page_config(
        page_title="🚀 Nuclear Sentiment Analyzer Pro",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🚀 Convosense360 Sentiment Analyzer
        </h1>
        <p style="color: white; text-align: center; margin: 0; opacity: 0.9;">
            Advanced Multi-Dimensional Sentiment Analysis with AI-Powered Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Standard", "Advanced", "Nuclear"],
            index=2
        )
        
        show_debug = st.checkbox("Show Debug Info", False)
        realtime_processing = st.checkbox("Real-time Processing", True)
        
        st.header("📊 Analysis Options")
        show_user_profiles = st.checkbox("User Personality Profiles", True)
        show_temporal_analysis = st.checkbox("Temporal Analysis", True)
        show_topic_modeling = st.checkbox("Topic Modeling", True)
        show_conversation_flow = st.checkbox("Conversation Flow", True)
    
    # File upload
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader(
        "Upload your chat data (CSV, TXT, or Excel)",
        type=['csv', 'txt', 'xlsx'],
        help="Ensure your file has 'message' and 'user' columns. Timestamp column is optional but recommended."
    )
    
    if uploaded_file:
        try:
            # Calculate file hash for caching
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Check if we need to reprocess
            if st.session_state.file_hash != file_hash:
                st.session_state.processed_data = None
                st.session_state.file_hash = file_hash
            
            if st.session_state.processed_data is None:
                with st.spinner("🔄 Loading and preprocessing data..."):
                    # Load data based on file type
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    else:  # txt file
                        content = uploaded_file.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        # Simple parsing for txt files
                        data_rows = []
                        for line in lines:
                            if ':' in line:
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    data_rows.append({'user': parts[0].strip(), 'message': parts[1].strip()})
                        df = pd.DataFrame(data_rows)
                    
                    # Validate and clean data
                    required_columns = ['message', 'user']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {missing_columns}")
                        st.stop()
                    
                    # Clean and prepare data
                    df = df.dropna(subset=['message', 'user'])
                    df = df[df['message'].str.len() > 0]
                    
                    # Parse timestamps if available
                    timestamp_columns = ['timestamp', 'time', 'date', 'datetime']
                    timestamp_col = None
                    
                    for col in timestamp_columns:
                        if col in df.columns:
                            timestamp_col = col
                            break
                    
                    if timestamp_col:
                        try:
                            df['timestamp'] = pd.to_datetime(df[timestamp_col])
                        except:
                            st.warning(f"Could not parse timestamp column '{timestamp_col}'. Proceeding without timestamps.")
                            df['timestamp'] = pd.NaT
                    else:
                        # Generate fake timestamps for demo purposes
                        start_time = datetime.now() - timedelta(hours=len(df))
                        df['timestamp'] = [start_time + timedelta(minutes=i*2) for i in range(len(df))]
                
                # Advanced processing
                st.success(f"✅ Loaded {len(df)} messages from {df['user'].nunique()} users")
                
                # Process with advanced sentiment analysis
                messages = df['message'].tolist()
                users = df['user'].tolist()
                timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else None
                
                results = advanced_parallel_processing(messages, users, timestamps)
                
                if results:
                    processed_df = pd.DataFrame(results)
                    st.session_state.processed_data = processed_df
                else:
                    st.error("Failed to process data")
                    st.stop()
            
            # Use cached processed data
            data = st.session_state.processed_data.copy()
            
            # Main dashboard
            st.header("📈 Analysis Dashboard")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                avg_sentiment = data['compound'].mean()
                st.metric(
                    "Overall Sentiment",
                    f"{avg_sentiment:.3f}",
                    delta=f"{avg_sentiment - 0:.3f}" if avg_sentiment != 0 else None
                )
            
            with col2:
                positive_ratio = (data['compound'] > 0.1).mean()
                st.metric("Positive Messages", f"{positive_ratio:.1%}")
            
            with col3:
                negative_ratio = (data['compound'] < -0.1).mean()
                st.metric("Negative Messages", f"{negative_ratio:.1%}")
            
            with col4:
                avg_confidence = data['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            with col5:
                total_users = data['user'].nunique()
                st.metric("Active Users", total_users)
            
            # Advanced user analysis
            if show_user_profiles:
                user_analysis = perform_advanced_user_analysis(data)
                st.session_state.user_profiles = user_analysis
            else:
                user_analysis = {}
            
            # Create advanced visualizations
            create_advanced_visualizations(data, user_analysis)
            
            # Additional analysis modules
            if show_conversation_flow and 'timestamp' in data.columns:
                create_conversation_flow_analysis(data)
            
            if show_topic_modeling:
                create_topic_modeling_analysis(data)
            
            # AI-powered insights
            create_sentiment_insights_dashboard(data, user_analysis)
            
            # Debug information
            if show_debug:
                st.header("🔧 Debug Information")
                st.write("**Data Shape:**", data.shape)
                st.write("**Columns:**", data.columns.tolist())
                st.write("**Sample Data:**")
                st.dataframe(data.head())
                
                if user_analysis:
                    st.write("**User Analysis:**")
                    st.json(user_analysis)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            if show_debug:
                st.exception(e)
    
    else:
        # Demo section
        st.header("🎯 Quick Demo")
        st.write("Try the analyzer with sample messages:")
        
        demo_messages = [
            "I absolutely love this new feature! 🚀",
            "This is terrible and broken 😡",
            "Not sure about this... seems okay I guess 🤔",
            "AMAZING work everyone! You're the best! ❤️🎉",
            "Why doesn't this work? So frustrating! 😤"
        ]
        
        selected_demo = st.selectbox("Choose a demo message:", demo_messages)
        
        if st.button("Analyze Demo Message"):
            engine = get_advanced_sentiment_engine()
            result = engine.nuclear_sentiment_analysis(selected_demo)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Analysis Results:**")
                st.write(f"• Compound Score: {result['compound']:.3f}")
                st.write(f"• Positive: {result['pos']:.3f}")
                st.write(f"• Negative: {result['neg']:.3f}")
                st.write(f"• Neutral: {result['neu']:.3f}")
                st.write(f"• Confidence: {result['confidence']:.3f}")
            
            with col2:
                st.write("**Context Analysis:**")
                if result.get('context', {}).get('detected_contexts'):
                    st.write("• Contexts:", ', '.join(result['context']['detected_contexts']))
                if result.get('context', {}).get('emotions'):
                    st.write("• Emotions:", ', '.join(result['context']['emotions']))
                
                st.write("**Components:**")
                for component, score in result.get('components', {}).items():
                    st.write(f"• {component}: {score:.3f}")

if __name__ == "__main__":
    main()