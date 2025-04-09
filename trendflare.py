import pandas as pd
import plotly.express as px
import streamlit as st
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import chardet
import io
from typing import Any, Dict

# For model evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------- Configuration -------------------
CONFIG = {
    "data_dir": "data",             # Directory containing country-specific data files.
    "date_format": "%y.%d.%m",       # Expected date format in the CSV files.
    "supported_countries": {        # Hardcoded country mapping.
        'Canada': 'CA',
        'Germany': 'DE',
        'France': 'FR',
        'India': 'IN',
        'Japan': 'JP',
        'South Korea': 'KR',
        'Mexico': 'MX',
        'Russia': 'RU',
        'United States': 'US'
    }
}

# ------------------- Initialization -------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# =============================================================================
# DATA LOADING & CACHING FUNCTIONS
# =============================================================================
def detect_encoding_fileobj(file_obj: Any, num_bytes: int = 10000) -> str:
    """Detect the encoding of a file-like object using chardet."""
    try:
        sample = file_obj.read(num_bytes)
        file_obj.seek(0)
        result = chardet.detect(sample)
        encoding = result.get('encoding')
        return encoding if encoding else 'windows-1252'
    except Exception as e:
        st.error(f"Error detecting encoding: {e}")
        return 'windows-1252'

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file: Any) -> pd.DataFrame:
    """Load CSV from a file-like object using detected encoding."""
    try:
        encoding = detect_encoding_fileobj(uploaded_file)
        content = uploaded_file.read().decode(encoding, errors="replace")
        uploaded_file.seek(0)
        df = pd.read_csv(io.StringIO(content))
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_csv_from_path(data_path: str) -> pd.DataFrame:
    """Load CSV from a file path using detected encoding."""
    try:
        with open(data_path, 'rb') as f:
            sample = f.read(10000)
            result = chardet.detect(sample)
            encoding = result.get('encoding') or 'windows-1252'
        with open(data_path, 'r', encoding=encoding, errors="replace") as f:
            df = pd.read_csv(f)
        return df
    except Exception as e:
        st.error(f"Error reading CSV from path: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_data_from_file(uploaded_file: Any) -> pd.DataFrame:
    """Load CSV file from an uploaded file with proper encoding handling."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower().strip()
        if file_extension == 'csv':
            return load_csv(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_data_for_country(country_name: str) -> pd.DataFrame:
    """
    Load CSV data for a specific country using a predefined mapping.
    The data file is assumed to be in the directory defined in CONFIG.
    """
    country_code = CONFIG["supported_countries"].get(country_name)
    if not country_code:
        st.error(f"No country code found for {country_name}.")
        return None
    data_path = os.path.join(CONFIG["data_dir"], f"{country_code}videos.csv")
    if not os.path.exists(data_path):
        st.error(f"The file for {country_name} ({country_code}) does not exist at {data_path}.")
        return None
    return load_csv_from_path(data_path)

@st.cache_data(show_spinner=False)
def load_category_mapping() -> dict:
    """Return a mapping of category IDs to category names."""
    return {
        1: "Film & Animation",
        2: "Autos & Vehicles",
        10: "Music",
        15: "Pets & Animals",
        17: "Sports",
        19: "Travel & Events",
        20: "Gaming",
        22: "People & Blogs",
        23: "Comedy",
        24: "Entertainment",
        25: "News & Politics",
        26: "Howto & Style",
        27: "Education",
        28: "Science & Technology",
        29: "Nonprofits & Activism",
    }

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def extract_tags_as_hashtags(tag_str: str) -> str:
    """Extract hashtags from the 'tags' column by splitting, cleaning, and prefixing with '#'."""
    if not tag_str or pd.isna(tag_str):
        return 'No Hashtags'
    tags = re.split(r'\|', tag_str)
    tags = [tag.strip().strip('"') for tag in tags if tag.strip() and tag.strip() != '[none]']
    hashtags = ['#' + tag if not tag.startswith('#') else tag for tag in tags]
    return ', '.join(hashtags) if hashtags else 'No Hashtags'

def clean_data(df: pd.DataFrame, category_mapping: dict) -> pd.DataFrame:
    """
    Clean the dataset by ensuring necessary columns exist, converting types,
    parsing dates (using the configurable date format), mapping categories, 
    and removing duplicates.
    """
    st.write(f"Initial data shape: {df.shape}")
    necessary_columns = ['views', 'likes', 'dislikes', 'trending_date', 'title', 'category_id']
    missing_cols = []
    for col in necessary_columns:
        if col not in df.columns:
            missing_cols.append(col)
            if col == 'trending_date':
                df[col] = pd.to_datetime('today')
            elif col == 'title':
                df[col] = 'Unknown Title'
            elif col in ['views', 'likes', 'dislikes']:
                df[col] = 0
            elif col == 'category_id':
                df[col] = -1
        else:
            if col in ['views', 'likes', 'dislikes']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            elif col == 'trending_date':
                df[col] = pd.to_datetime(df[col], format=CONFIG["date_format"], errors='coerce')
    if missing_cols:
        st.warning(f"Missing columns: {missing_cols}. Default values were assigned.")
    df = df.dropna(subset=['title', 'trending_date'])
    df = df.drop_duplicates(subset=['title'], keep='first')
    if 'category_id' in df.columns:
        df['category_name'] = df['category_id'].apply(lambda x: category_mapping.get(x, "Unknown Category"))
    if 'category_name' in df.columns and df['category_name'].isnull().sum() > 0:
        st.warning("Some rows have missing/unknown categories.")
    st.write(f"Cleaned data shape: {df.shape}")
    return df

# -----------------------------------------------------------------------------
# SENTIMENT ANALYSIS FUNCTIONS (VADER Only)
# -----------------------------------------------------------------------------
def perform_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sentiment scores using VADER by combining title and description.
    Applies VADER's recommended thresholds:
        - Compound score >= 0.05 -> Positive
        - Compound score <= -0.05 -> Negative
        - Otherwise, Neutral
    Also extracts hashtags.
    """
    def get_sentiment_label(score: float) -> str:
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df['title'] = df['title'].astype(str).str.strip()
    if 'description' in df.columns:
        df['description'] = df['description'].astype(str).str.strip()
    else:
        df['description'] = ""
    df['combined_text'] = (df['title'] + " " + df['description']).str.lower().str.strip()
    df['sentiment_score'] = df['combined_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(get_sentiment_label)
    if 'tags' in df.columns:
        df['hashtags'] = df['tags'].apply(extract_tags_as_hashtags)
    else:
        df['hashtags'] = 'No Hashtags'
    return df

# -----------------------------------------------------------------------------
# MODEL EVALUATION FUNCTIONS
# -----------------------------------------------------------------------------
def evaluate_sentiment_model(df: pd.DataFrame) -> None:
    """
    Evaluate the sentiment analysis model if ground truth labels are available.
    Expects a 'true_sentiment' column in the DataFrame.
    """
    if 'true_sentiment' not in df.columns:
        st.warning("No ground truth sentiment labels found in the dataset.")
        return

    y_true = df['true_sentiment']
    y_pred = df['sentiment']
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=["Positive", "Neutral", "Negative"])
    
    st.subheader("Sentiment Model Evaluation Metrics 📊")
    st.write(f"Accuracy: {acc*100:.2f}% ✅")
    st.write(f"Precision: {prec*100:.2f}%")
    st.write(f"Recall: {rec*100:.2f}%")
    st.write(f"F1-Score: {f1*100:.2f}%")
    
    fig_cm = px.imshow(conf_mat,
                       labels=dict(x="Predicted", y="True", color="Count"),
                       x=["Positive", "Neutral", "Negative"],
                       y=["Positive", "Neutral", "Negative"],
                       text_auto=True,
                       title="Confusion Matrix 😮")
    # Force empty title for the figure so nothing extra is shown.
    fig_cm.update_layout(title_text="")
    st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------------------------------------------------------
# OTHER ANALYSIS FUNCTIONS (Video statistics, etc.)
# -----------------------------------------------------------------------------
def get_top_hashtags(df: pd.DataFrame, n: int = 10):
    if 'hashtags' not in df.columns:
        st.warning("No hashtags column found in dataset.")
        return None
    df['hashtags'] = df['hashtags'].fillna('')
    all_hashtags = df['hashtags'].str.split(',').explode().str.strip()
    all_hashtags = all_hashtags[(all_hashtags != '') & (~all_hashtags.str.lower().eq('no hashtags'))]
    return all_hashtags.value_counts().head(n)

def get_top_channels_by_views(df: pd.DataFrame, n: int = 10):
    if 'channel_title' not in df.columns:
        st.warning("No 'channel_title' column found in dataset.")
        return None
    channels = df.groupby('channel_title')['views'].sum().reset_index()
    return channels.nlargest(n, 'views')

def get_top_channels_by_videos(df: pd.DataFrame, n: int = 10):
    if 'channel_title' not in df.columns:
        st.warning("No 'channel_title' column found in dataset.")
        return None
    channel_counts = df['channel_title'].value_counts().reset_index()
    channel_counts.columns = ['channel_title', 'video_count']
    return channel_counts.head(n)

def get_top_liked_videos(df: pd.DataFrame, n: int = 10):
    if 'likes' not in df.columns:
        st.error("'likes' column is missing.")
        return None
    return df.nlargest(n, 'likes')[['title', 'likes']]

def get_top_disliked_videos(df: pd.DataFrame, n: int = 10):
    if 'dislikes' not in df.columns:
        st.error("'dislikes' column is missing.")
        return None
    return df.nlargest(n, 'dislikes')[['title', 'dislikes']]

def get_top_commented_videos(df: pd.DataFrame, n: int = 10):
    if 'comment_count' not in df.columns:
        df['comment_count'] = 0
        st.warning("'comment_count' column missing; defaulting to 0.")
    top_commented = df.groupby('title')['comment_count'].sum().reset_index()
    return top_commented.nlargest(n, 'comment_count')[['title', 'comment_count']]

@st.cache_data(show_spinner=False)
def filter_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    return df[(df['trending_date'] >= start_date) & (df['trending_date'] <= end_date)]

# -----------------------------------------------------------------------------
# CHART CREATION FUNCTIONS
# -----------------------------------------------------------------------------
# All internal chart titles are set to an empty string so that only the st.subheader
# headings are displayed.

def create_interactive_barplot(df: pd.DataFrame, n: int, chart_title: str = ""):
    if not {"views", "title"}.issubset(df.columns):
        st.error("Required columns for bar plot are missing.")
        return px.bar(title="", template='plotly_white')
    df_clean = df.dropna(subset=['views', 'title'])
    top_videos = df_clean[['title', 'views']].nlargest(n, 'views').sort_values(by='views', ascending=False)
    fig = px.bar(
        top_videos, 
        x='views', 
        y='title', 
        title="",
        labels={'title': 'Video Title', 'views': 'Views'}, 
        color='views', 
        color_continuous_scale='Blues', 
        text='views', 
        orientation='h',
        template='plotly_white'
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        margin={'l': 200, 'r': 50, 't': 50, 'b': 50},
        xaxis_tickformat=".0f",
        font=dict(family="Helvetica", size=14),
        title_text=""
    )
    return fig

def create_sentiment_distribution_chart(df: pd.DataFrame, chart_title: str = ""):
    if "sentiment" not in df.columns:
        st.error("Sentiment column is missing.")
        return px.pie(title="", template='plotly_white')
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts,
        title="",
        color=sentiment_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
        template='plotly_white'
    )
    fig.update_layout(font=dict(family="Helvetica", size=14), title_text="")
    return fig

def create_category_distribution_chart(df: pd.DataFrame, chart_title: str = ""):
    if "category_name" not in df.columns:
        st.error("Category name column is missing.")
        return px.pie(title="", template='plotly_white')
    category_counts = df['category_name'].value_counts()
    fig = px.pie(
        names=category_counts.index,
        values=category_counts,
        title="",
        color=category_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
        template='plotly_white'
    )
    fig.update_layout(font=dict(family="Helvetica", size=14), title_text="")
    return fig

def create_interactive_monthly_trend_plot(df: pd.DataFrame, chart_title: str = ""):
    if "trending_date" not in df.columns:
        st.error("Trending date column is missing.")
        return px.line(title="", template='plotly_white')
    df['trending_date'] = pd.to_datetime(df['trending_date'], errors='coerce')
    df = df.dropna(subset=['trending_date'])
    df['Month_dt'] = df['trending_date'].dt.to_period('M').dt.to_timestamp()
    df_grouped = df.groupby('Month_dt')['views'].sum().reset_index().rename(columns={'views': 'Total Views'})
    df_grouped = df_grouped.sort_values('Month_dt')
    fig = px.line(
        df_grouped,
        x='Month_dt',
        y='Total Views',
        title="",
        labels={'Month_dt': 'Month', 'Total Views': 'Total Views'},
        markers=True,
        line_shape='linear',
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Total Views',
        hovermode='x unified',
        xaxis=dict(
            showgrid=True,
            tickformat="%Y-%m",
            dtick="M1"
        ),
        yaxis=dict(showgrid=True),
        plot_bgcolor='white',
        font=dict(family="Helvetica", size=14),
        title_text=""
    )
    return fig

def create_top_hashtags_bar_chart(df: pd.DataFrame, n: int = 10, chart_title: str = ""):
    top_tags = get_top_hashtags(df, n)
    if top_tags is None or top_tags.empty:
        return px.bar(title="", template='plotly_white')
    fig = px.bar(
        x=top_tags.values,
        y=top_tags.index,
        orientation='h',
        title="",
        labels={'x': 'Count', 'y': 'Hashtag'},
        color=top_tags.values,
        color_continuous_scale='Plasma',
        template='plotly_white'
    )
    fig.update_traces(text=top_tags.values, textposition='outside')
    fig.update_layout(
        margin={'l': 150, 'r': 50, 't': 50, 'b': 50},
        font=dict(family="Helvetica", size=14),
        title_text=""
    )
    return fig

def create_top_liked_videos_bar_chart(df: pd.DataFrame, n: int = 10, chart_title: str = ""):
    top_liked = get_top_liked_videos(df, n)
    if top_liked is None or top_liked.empty:
        return px.bar(title="", template='plotly_white')
    fig = px.bar(
        top_liked,
        x='likes',
        y='title',
        title="",
        labels={'title': 'Video Title', 'likes': 'Likes'},
        color='likes',
        color_continuous_scale='Viridis',
        orientation='h',
        text='likes',
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_title='Likes',
        yaxis_title='Video Title',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        plot_bgcolor='white',
        font=dict(family="Helvetica", size=14),
        title_text=""
    )
    return fig

# =============================================================================
# STREAMLIT APP (UI) MAIN FUNCTION
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="VIDAS: Video Interactive Data Analytics and Sentiment Dashboard", layout="wide")
    st.markdown(
        """
        <style>
        body {
            background-color: #f7f7f7;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: left;
            background: linear-gradient(90deg, #6A82FB, #FC5C7D);
            -webkit-background-clip: text;
            color: transparent;
            text-shadow: 4px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(180deg, #ECE9E6, #FFFFFF);
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.markdown('<div class="title">VIDAS: Video Interactive Data Analytics and Sentiment Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to **VIDAS** – an interactive dashboard for analyzing video trends.
    - **Upload or Select Data:** Upload your CSV file or select a country to load pre-collected data.
    - **Data Processing:** Our system cleans your data, performs sentiment analysis on both title and description, and extracts useful features including channel and hashtag insights.
    - **Visualizations:** Explore trends with interactive charts and advanced channel analysis.
    - **Download:** Once processed, download the final CSV for further analysis.
    """)
    
    st.sidebar.header("Navigation")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    country_names = list(CONFIG["supported_countries"].keys())
    country_name = st.sidebar.selectbox("Or Select Country:", country_names)
    
    with st.spinner("Loading data..."):
        if uploaded_file is not None:
            df = load_data_from_file(uploaded_file)
        else:
            df = load_data_for_country(country_name)
    
    if df is None:
        st.error("Data could not be loaded. Please check the file and try again.")
        return
    
    category_mapping = load_category_mapping()
    with st.spinner("Cleaning data..."):
        df = clean_data(df, category_mapping)
    with st.spinner("Performing sentiment analysis using VADER..."):
        df = perform_sentiment_analysis(df)
    
    st.write(f"**Total valid videos:** {len(df)}")
    st.subheader("Dataset Summary")
    st.write(f"**Unique categories:** {df['category_name'].nunique()}")
    if df['trending_date'].notna().sum() > 0:
        st.write(f"**Date range:** {df['trending_date'].min().date()} to {df['trending_date'].max().date()}")
    else:
        st.write("No valid trending dates found.")
    
    if "true_sentiment" in df.columns:
        evaluate_sentiment_model(df)
    
    min_date = df['trending_date'].min().date()
    max_date = df['trending_date'].max().date()
    st.sidebar.markdown(f"**Data available from:** {min_date} to {max_date}")
    date_range = st.sidebar.date_input(
        "Select Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = filter_by_date(df, start_date, end_date)
        st.write(f"**Filtered videos:** from {date_range[0]} to {date_range[1]}")
    else:
        st.warning("Please select both a start and end date. Displaying all data.")
    
    st.sidebar.subheader("Dataset Columns")
    st.sidebar.write(df.columns.tolist())
    
    # --- N-based charts (these five charts use the slider value) ---
    top_n = st.slider("Select number of top items to display (N value)", min_value=5, max_value=20, value=10)
    
    st.subheader(f"Top {top_n} Videos by Views")
    fig_bar = create_interactive_barplot(df, top_n)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader(f"Sentiment Distribution for Top {top_n} Videos")
    top_videos = df.nlargest(top_n, 'views')
    fig_sentiment = create_sentiment_distribution_chart(top_videos)
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    st.subheader(f"Category Distribution for Top {top_n} Videos")
    fig_category_top = create_category_distribution_chart(top_videos)
    st.plotly_chart(fig_category_top, use_container_width=True)
    
    st.subheader(f"Top {top_n} Channels by Total Views")
    top_channels_views = get_top_channels_by_views(df, n=top_n)
    if top_channels_views is not None and not top_channels_views.empty:
        fig_channels_views = px.bar(
            top_channels_views,
            x='views',
            y='channel_title',
            orientation='h',
            title="",
            labels={'channel_title': 'Channel', 'views': 'Total Views'},
            color='views',
            color_continuous_scale='Blues',
            template='plotly_white'
        )
        fig_channels_views.update_layout(
            font=dict(family="Helvetica", size=14),
            title_text=""
        )
        st.plotly_chart(fig_channels_views, use_container_width=True)
    else:
        st.warning("Channel analysis not available for total views.")
    
    st.subheader(f"Top {top_n} Channels by Video Count")
    top_channels_videos = get_top_channels_by_videos(df, n=top_n)
    if top_channels_videos is not None and not top_channels_videos.empty:
        fig_channels_videos = px.bar(
            top_channels_videos,
            x='video_count',
            y='channel_title',
            orientation='h',
            title="",
            labels={'channel_title': 'Channel', 'video_count': 'Video Count'},
            color='video_count',
            color_continuous_scale='Oranges',
            template='plotly_white'
        )
        fig_channels_videos.update_layout(
            font=dict(family="Helvetica", size=14),
            title_text=""
        )
        st.plotly_chart(fig_channels_videos, use_container_width=True)
    else:
        st.warning("Channel analysis not available for video count.")
    
    # --- Other visualizations using fixed defaults (10 items) ---
    st.subheader("Top 10 Hashtags from Video Tags")
    with st.spinner("Generating top hashtags chart..."):
        fig_hashtags = create_top_hashtags_bar_chart(df, n=10)
        st.plotly_chart(fig_hashtags, use_container_width=True)
    
    st.subheader("Top 10 Videos by Likes")
    with st.spinner("Generating top liked videos chart..."):
        fig_top_liked = create_top_liked_videos_bar_chart(df, n=10)
        st.plotly_chart(fig_top_liked, use_container_width=True)
    
    st.subheader("Top 10 Videos by Dislikes")
    with st.spinner("Generating top disliked videos chart..."):
        top_disliked = get_top_disliked_videos(df, n=10)
        fig_top_disliked = px.bar(
            top_disliked,
            x='dislikes',
            y='title',
            title="",
            labels={'title': 'Video Title', 'dislikes': 'Dislikes'},
            color='dislikes',
            orientation='h',
            text='dislikes',
            color_continuous_scale='Reds',
            template='plotly_white'
        )
        fig_top_disliked.update_layout(
            margin={'l': 200, 'r': 50, 't': 50, 'b': 50},
            font=dict(family="Helvetica", size=14),
            title_text=""
        )
        st.plotly_chart(fig_top_disliked, use_container_width=True)
    
    st.subheader("Top 10 Videos by Comments")
    with st.spinner("Generating top commented videos chart..."):
        top_commented = get_top_commented_videos(df, n=10)
        fig_top_commented = px.bar(
            top_commented,
            x='comment_count',
            y='title',
            title="",
            labels={'title': 'Video Title', 'comment_count': 'Comments'},
            color='comment_count',
            orientation='h',
            text='comment_count',
            color_continuous_scale='Purples',
            template='plotly_white'
        )
        fig_top_commented.update_layout(
            margin={'l': 200, 'r': 50, 't': 50, 'b': 50},
            font=dict(family="Helvetica", size=14),
            title_text=""
        )
        st.plotly_chart(fig_top_commented, use_container_width=True)
    
    # --- Visualizations based on entire dataset ---
    st.subheader("Overall Category Distribution of All Videos")
    with st.spinner("Generating overall category distribution chart..."):
        fig_category_all = create_category_distribution_chart(df, "Overall Category Distribution of All Videos")
        st.plotly_chart(fig_category_all, use_container_width=True)
    
    st.subheader("Monthly View Trends for All Videos")
    with st.spinner("Generating monthly view trends chart..."):
        fig_monthly = create_interactive_monthly_trend_plot(df, "Monthly View Trends for All Videos")
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    st.subheader("Download Processed CSV")
    with st.spinner("Preparing download..."):
        if not df.empty:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Final Processed CSV",
                data=csv_data,
                file_name="processed_videos.csv",
                mime="text/csv"
            )
        else:
            st.error("No data available for download.")

if __name__ == "__main__":
    main()
