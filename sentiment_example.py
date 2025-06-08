import streamlit as st
import sentiment_helper
import sentiment_preprocessor
import pandas as pd
from googletrans import Translator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures
import time

# Initialize session state variables
if "show_sentiment" not in st.session_state:
    st.session_state.show_sentiment = True  # Set the default value, True or False based on your requirement

# Check if sentiment section should be shown
if st.session_state.show_sentiment:
    st.title("Sentiment Analysis")

    # Decode uploaded WhatsApp chat file
    uploaded_file = st.file_uploader("Upload WhatsApp Chat", type=["txt"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        df = bytes_data.decode("utf-8")
        data = sentiment_preprocessor.Preprocess(df)

        # Setup translator and sentiment analyzer
        translator = Translator()
        sentiments = SentimentIntensityAnalyzer()

        # Function to safely translate messages
        def safe_translate(msg):
            try:
                if isinstance(msg, str) and msg.strip():
                    return translator.translate(msg).text
                return msg
            except Exception as e:
                print(f"Error in translation: {e}")
                return msg

        # Function to apply sentiment analysis
        def apply_sentiment(msg):
            try:
                sentiment_scores = sentiments.polarity_scores(msg)
                return sentiment_scores["pos"], sentiment_scores["neg"], sentiment_scores["neu"]
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                return 0, 0, 1  # Neutral sentiment if error

        # Add progress bar for user experience
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Parallelize the translation and sentiment analysis
        def process_messages(data_chunk):
            results = []
            for msg in data_chunk:
                # Translate Hinglish to English
                translated_msg = safe_translate(msg)
                # Apply sentiment analysis
                pos, neg, neu = apply_sentiment(translated_msg)
                results.append((translated_msg, pos, neg, neu))
            return results

        # Function to process data in chunks using concurrent futures
        def parallel_process_data(data, chunk_size=500):
            num_chunks = (len(data) // chunk_size) + 1
            all_results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size
                    chunk = data[start:end]
                    futures.append(executor.submit(process_messages, chunk))
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    progress_bar.progress((i + 1) / num_chunks)
                    progress_text.text(f"Processing chunk {i + 1}/{num_chunks}...")

            return all_results

        if st.button("Show Analysis"):
            # Process messages in chunks
            st.write("Processing messages...")
            start_time = time.time()
            data_chunked = parallel_process_data(data['message'])
            data[['translated', 'po', 'ne', 'nu']] = pd.DataFrame(data_chunked, columns=['translated', 'po', 'ne', 'nu'])

            st.write(f"Processing completed in {round(time.time() - start_time, 2)} seconds.")
            
            # Apply sentiment scoring (Positive, Neutral, Negative)
            def sentiment_score(row):
                if row["po"] > 0.1:  # Positive sentiment
                    return 1
                elif row["ne"] > 0.1:  # Negative sentiment
                    return -1
                return 0  # Neutral

            data['value'] = data.apply(sentiment_score, axis=1)

            # Sidebar user selection
            user_list = data['user'].unique().tolist()
            user_list.sort()
            user_list.insert(0, "Overall")
            selected_user = st.sidebar.selectbox("Show analysis wrt", user_list, key="sentiment_analysis_user_selector")

            if selected_user == "Overall":
                # Show the sentiment analysis data
                st.write("### Sentiment Analysis Results")
                st.dataframe(data[['user', 'message', 'translated', 'po', 'ne', 'nu', 'value']])

            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                # Monthly Activity
                st.markdown(f"### Monthly Activity Map ({label})")
                busy_month = sentiment_helper.month_activity_map(selected_user, data, sentiment_value)
                busy_month = busy_month.reset_index()
                busy_month.columns = ['month', 'count']
                fig = px.bar(busy_month, x='month', y='count', title=f"Monthly Activity - {label}", color_discrete_sequence=[color])
                st.plotly_chart(fig)

                # Daily Activity
                st.markdown(f"### Daily Activity Map ({label})")
                busy_day = sentiment_helper.week_activity_map(selected_user, data, sentiment_value)
                busy_day = busy_day.reset_index()
                busy_day.columns = ['day', 'count']
                fig = px.bar(busy_day, x='day', y='count', title=f"Daily Activity - {label}", color_discrete_sequence=[color])
                st.plotly_chart(fig)

            # Weekly Heatmap
            for label, sentiment_value in zip(["Positive", "Neutral", "Negative"], [1, 0, -1]):
                try:
                    st.markdown(f"### Weekly Activity Heatmap ({label})")
                    heatmap = sentiment_helper.activity_heatmap(selected_user, data, sentiment_value)
                    fig = go.Figure(data=go.Heatmap(z=heatmap.values, x=heatmap.columns, y=heatmap.index, colorscale='Viridis'))
                    fig.update_layout(title=f"Weekly Heatmap - {label}")
                    st.plotly_chart(fig)
                except Exception as e:
                    print(f"Error in heatmap generation: {e}")
                    st.image('error.webp')

            # Timelines
            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                st.markdown(f"### Daily Timeline ({label})")
                timeline_daily = sentiment_helper.daily_timeline(selected_user, data, sentiment_value)
                fig = px.line(timeline_daily, x='only_date', y='message', title=f"Daily Timeline - {label}", markers=True, color_discrete_sequence=[color])
                st.plotly_chart(fig)

                st.markdown(f"### Monthly Timeline ({label})")
                timeline_monthly = sentiment_helper.monthly_timeline(selected_user, data, sentiment_value)
                fig = px.line(timeline_monthly, x='time', y='message', title=f"Monthly Timeline - {label}", markers=True, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            if selected_user == "Overall":
                for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                    # Contribution Table
                    st.markdown(f"### Most {label} Contribution")
                    df_percent = sentiment_helper.percentage(data, sentiment_value)
                    st.dataframe(df_percent)

                    # Top Users
                    st.markdown(f"### Most {label} Users")
                    user_counts = data['user'][data['value'] == sentiment_value].value_counts().head(10)
                    user_counts_df = user_counts.reset_index()
                    user_counts_df.columns = ['user', 'count']
                    fig = px.bar(user_counts_df, x='user', y='count', labels={'user': 'User', 'count': 'Messages'}, 
                                title=f"Top {label} Users", color_discrete_sequence=[color])
                    st.plotly_chart(fig)

            # Word Clouds & Common Words
            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                try:
                    st.markdown(f"### {label} Word Cloud")
                    wordcloud = sentiment_helper.create_wordcloud(selected_user, data, sentiment_value)
                    st.image(wordcloud.to_array())
                except Exception as e:
                    print(f"Error generating word cloud: {e}")
                    st.image('error.webp')

                try:
                    st.markdown(f"### {label} Words")
                    common_words = sentiment_helper.most_common_words(selected_user, data, sentiment_value)
                    fig = px.bar(common_words, x=1, y=0, orientation='h', 
                                labels={"0": "Words", "1": "Count"}, title=f"Most Common {label} Words", 
                                color_discrete_sequence=[color])
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)
                except Exception as e:
                    print(f"Error generating common words: {e}")
                    st.image('error.webp')
