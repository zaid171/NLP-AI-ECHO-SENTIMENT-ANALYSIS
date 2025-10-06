import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import plotly.express as px
import os

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()

class DataPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        if text is None or text != text:  # Check for NaN
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

class SentimentAnalyzerApp:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.vectorizer = None
        self.df = None
        
    def load_sample_data(self):
        """Create sample data for demo purposes"""
        try:
            sample_data = {
                'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
                'review': [
                    'This app is absolutely amazing and very helpful!',
                    'The application works okay but could be better.',
                    'I am very disappointed with the performance.',
                    'Excellent features and great user interface.',
                    'Not what I expected, needs improvement.'
                ],
                'rating': [5, 3, 1, 5, 2],
                'platform': ['Web', 'Mobile', 'Web', 'Mobile', 'Web'],
                'language': ['en', 'en', 'en', 'en', 'en'],
                'location': ['USA', 'UK', 'Canada', 'Australia', 'India'],
                'verified_purchase': ['Yes', 'No', 'Yes', 'Yes', 'No'],
                'helpful_votes': [10, 2, 5, 8, 1]
            }
            self.df = pd.DataFrame(sample_data)
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Create sentiment labels
            def get_sentiment(rating):
                if rating >= 4:
                    return 'Positive'
                elif rating == 3:
                    return 'Neutral'
                else:
                    return 'Negative'
            
            self.df['sentiment'] = self.df['rating'].apply(get_sentiment)
            return True
        except Exception as e:
            st.error(f"Error creating sample data: {e}")
            return False
    
    def load_real_data(self):
        """Try to load real data from file"""
        try:
            data_path = 'data/chatgpt_style_reviews_dataset.csv'
            if os.path.exists(data_path):
                self.df = pd.read_csv(data_path)
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
                
                # Create sentiment labels
                def get_sentiment(rating):
                    if rating >= 4:
                        return 'Positive'
                    elif rating == 3:
                        return 'Neutral'
                    else:
                        return 'Negative'
                
                self.df['sentiment'] = self.df['rating'].apply(get_sentiment)
                return True
            return False
        except Exception as e:
            st.error(f"Error loading real data: {e}")
            return False
    
    def load_model(self):
        """Try to load model, but use simulated predictions if not available"""
        try:
            model_path = 'models/sentiment_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.vectorizer = model_data['vectorizer']
                return True
            else:
                st.info("🤖 Using simulated sentiment analysis for demo. Upload a trained model for accurate predictions.")
                return False
        except Exception as e:
            st.warning(f"Model loading failed: {e}. Using simulated mode.")
            return False
    
    def ensure_data_loaded(self):
        """Ensure data is loaded, use sample if real data not available"""
        if self.df is None:
            # First try to load real data
            if not self.load_real_data():
                # If real data fails, load sample data
                self.load_sample_data()
    
    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if self.model is not None and self.vectorizer is not None:
            # Use actual model
            cleaned_text = self.preprocessor.clean_text(text)
            processed_text = self.preprocessor.tokenize_and_lemmatize(cleaned_text)
            text_vector = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_vector)[0]
            probability = self.model.predict_proba(text_vector)[0]
            return prediction, dict(zip(self.model.classes_, probability))
        else:
            # Simulate prediction
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'awesome', 'perfect', 'fantastic', 'wonderful', 'outstanding']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'disappointed', 'poor', 'horrible', 'waste', 'useless']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                prediction = "Positive"
                confidence = min(0.8 + (positive_count * 0.05), 0.95)
            elif negative_count > positive_count:
                prediction = "Negative" 
                confidence = min(0.8 + (negative_count * 0.05), 0.95)
            else:
                prediction = "Neutral"
                confidence = 0.6
            
            # Simulate probabilities
            if prediction == "Positive":
                probabilities = {'Positive': confidence, 'Neutral': (1-confidence)/2, 'Negative': (1-confidence)/2}
            elif prediction == "Negative":
                probabilities = {'Positive': (1-confidence)/2, 'Neutral': (1-confidence)/2, 'Negative': confidence}
            else:
                probabilities = {'Positive': 0.2, 'Neutral': confidence, 'Negative': 0.2}
            
            return prediction, probabilities
    
    def run(self):
        """Main application"""
        st.set_page_config(
            page_title="AI Echo - Sentiment Analysis",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">🤖 AI Echo: Sentiment Analysis</h1>', unsafe_allow_html=True)
        st.markdown("### Customer Review Sentiment Analysis Dashboard")
        
        # Initialize and load data
        self.ensure_data_loaded()
        
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = self.load_model()
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["📊 Overview", "🤖 Model Demo", "📈 Analysis", "💡 Insights"]
        )
        
        # Page routing
        if page == "📊 Overview":
            self.show_overview()
        elif page == "🤖 Model Demo":
            self.show_model_demo()
        elif page == "📈 Analysis":
            self.show_analysis()
        else:
            self.show_insights()
    
    def show_overview(self):
        """Overview page"""
        st.header("📊 Project Overview")
        
        # Ensure data is loaded
        self.ensure_data_loaded()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reviews = len(self.df)
            st.metric("Total Reviews", total_reviews)
        
        with col2:
            avg_rating = self.df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        
        with col3:
            positive_pct = (self.df['sentiment'] == 'Positive').mean() * 100
            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        
        with col4:
            helpful_reviews = self.df['helpful_votes'].sum()
            st.metric("Total Helpful Votes", helpful_reviews)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Review Rating Distribution")
            rating_counts = self.df['rating'].value_counts().sort_index()
            fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values,
                        labels={'x': 'Rating', 'y': 'Count'},
                        title='Distribution of Ratings')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Distribution")
            sentiment_counts = self.df['sentiment'].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                        title='Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show data source info
        if hasattr(self, 'using_real_data') and self.using_real_data:
            st.success("✅ Using real dataset from file")
        else:
            st.info("💡 Using sample data for demo. Upload your dataset to the 'data' folder for real analysis.")
    
    def show_model_demo(self):
        """Interactive model demo"""
        st.header("🤖 Sentiment Analysis Demo")
        
        st.markdown("""
        Enter your own review text below to analyze its sentiment.
        The model will predict whether the sentiment is **Positive**, **Neutral**, or **Negative**.
        """)
        
        # Text input
        user_text = st.text_area(
            "Enter your review text:",
            height=150,
            placeholder="Type your review here... Example: 'This app is amazing and very helpful!'",
            value="I love this application! It's incredibly useful and well-designed."
        )
        
        if user_text:
            with st.spinner("Analyzing sentiment..."):
                prediction, probabilities = self.predict_sentiment(user_text)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                sentiment_colors = {
                    'Positive': '🟢',
                    'Neutral': '🟡', 
                    'Negative': '🔴'
                }
                
                st.metric(
                    "Predicted Sentiment",
                    f"{sentiment_colors.get(prediction, '⚪')} {prediction}"
                )
            
            with col2:
                st.subheader("Confidence Scores")
                
                for sentiment, prob in probabilities.items():
                    st.write(f"**{sentiment}**: {prob:.1%}")
                    st.progress(prob)
            
            if self.model is None:
                st.info("🔬 Currently using simulated analysis. Upload a trained model file for more accurate predictions.")
        
        # Example reviews
        st.markdown("---")
        st.subheader("💡 Try these examples:")
        
        examples = [
            "This app is absolutely fantastic! It helps me so much with my work.",
            "The application is okay, but it could use some improvements.",
            "I'm very disappointed with the performance and customer service.",
            "Outstanding features and excellent user experience!",
            "It's mediocre, nothing special about it."
        ]
        
        cols = st.columns(3)
        for i, example in enumerate(examples):
            with cols[i % 3]:
                if st.button(f"'{example[:30]}...'", use_container_width=True):
                    st.rerun()

    def show_analysis(self):
        """Analysis page"""
        st.header("📈 Data Analysis")
        
        # Ensure data is loaded
        self.ensure_data_loaded()
        
        if self.df is None:
            st.error("No data available for analysis.")
            return
        
        # Platform analysis
        st.subheader("Platform Comparison")
        platform_counts = self.df['platform'].value_counts()
        fig = px.bar(platform_counts, x=platform_counts.index, y=platform_counts.values,
                    labels={'x': 'Platform', 'y': 'Number of Reviews'},
                    title='Reviews by Platform')
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment by platform
        platform_sentiment = pd.crosstab(self.df['platform'], self.df['sentiment'], normalize='index') * 100
        fig = px.bar(platform_sentiment, barmode='stack',
                    title='Sentiment Distribution by Platform (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Word clouds
        st.subheader("📝 Word Clouds")
        
        positive_text = ' '.join(self.df[self.df['sentiment'] == 'Positive']['review'])
        negative_text = ' '.join(self.df[self.df['sentiment'] == 'Negative']['review'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Reviews**")
            if positive_text.strip():
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(positive_text)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("No positive reviews available")
        
        with col2:
            st.markdown("**Negative Reviews**")
            if negative_text.strip():
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(negative_text)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info("No negative reviews available")

    def show_insights(self):
        """Insights page"""
        st.header("💡 Business Insights & Recommendations")
        
        # Ensure data is loaded
        self.ensure_data_loaded()
        
        if self.df is None:
            st.error("No data available for insights.")
            return
        
        # Key metrics
        positive_pct = (self.df['sentiment'] == 'Positive').mean() * 100
        avg_rating = self.df['rating'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Satisfaction", f"{positive_pct:.1f}%")
        
        with col2:
            st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
        
        with col3:
            verified_ratio = (self.df['verified_purchase'] == 'Yes').mean() * 100
            st.metric("Verified Reviews", f"{verified_ratio:.1f}%")
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("🎯 Actionable Recommendations")
        
        recommendations = [
            "**Monitor Negative Reviews**: Regularly analyze 1-2 star reviews for common issues and pain points",
            "**Platform Optimization**: Ensure consistent user experience across all platforms (Web, Mobile, etc.)",
            "**Feature Development**: Prioritize features frequently mentioned in positive reviews",
            "**Customer Support**: Implement sentiment-based routing for support tickets",
            "**Regional Strategy**: Analyze location-based sentiment for market-specific improvements",
            "**Version Tracking**: Monitor sentiment changes across different application versions"
        ]
        
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"{i}. {recommendation}")
        
        st.markdown("---")
        
        # Technical setup
        st.subheader("🔧 Technical Setup")
        st.info("""
        **To use with your own data:**
        1. Upload your CSV file to the `data/` folder
        2. Train and save your model as `models/sentiment_model.pkl`
        3. The app will automatically detect and use your files
        
        **Current mode:** Using sample data with simulated sentiment analysis
        """)

# Run the app
if __name__ == "__main__":
    app = SentimentAnalyzerApp()
    app.run()
