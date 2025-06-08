import nltk
import streamlit as st
import preprocessor
import helper
import chatbot
import re
import sentiment_preprocessor, sentiment_helper
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# for cohere api
import json
import os
from dotenv import load_dotenv
from chatbot import textual_analysis

# Load environment variables from .env file
load_dotenv()

# requirements for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    if "show_normal_analysis" not in st.session_state:
        st.session_state.show_normal_analysis = False

    if "show_summarization" not in st.session_state:
        st.session_state.show_summarization = False

    if "show_sentiment" not in st.session_state:
        st.session_state.show_sentiment = False

    if st.sidebar.button("Normal Analysis", key="normal_analysis_btn"):
        st.session_state.show_normal_analysis = True
        st.session_state.show_summarization = False
        st.session_state.show_sentiment = False

    if st.sidebar.button("Summarize Chat", key="summarize_btn"):
        st.session_state.show_normal_analysis = False
        st.session_state.show_summarization = True
        st.session_state.show_sentiment = False

    if st.sidebar.button("Sentiment Analysis", key="sentiment_analysis_btn"):
        st.session_state.show_normal_analysis = False
        st.session_state.show_summarization = False
        st.session_state.show_sentiment = True

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    if st.session_state.show_normal_analysis:
        selected_user = st.sidebar.selectbox("Show analysis wrt", user_list, key="normal_analysis_user_selector")


# ====================================== NORML CHAT ANALYSIS =============================================================
    if st.session_state.show_normal_analysis:
        st.title("Normal Chat Analysis")

        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.subheader("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Messages", num_messages)
        with col2:
            st.metric("Words", words)
        with col3:
            st.metric("Media", num_media_messages)
        with col4:
            st.metric("Links", num_links)

        st.subheader("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig = px.line(timeline, x='time', y='message', title='Monthly Timeline', markers=True)
        st.plotly_chart(fig)

        st.subheader("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig = px.line(daily_timeline, x='only_date', y='message', title='Daily Timeline', markers=True)
        st.plotly_chart(fig)

        st.subheader("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Most Busy Day**")
            busy_day = helper.week_activity_map(selected_user, df)
            fig = px.bar(x=busy_day.index, y=busy_day.values, labels={'x': 'Day', 'y': 'Messages'}, title='Busy Day')
            st.plotly_chart(fig)

        with col2:
            st.markdown("**Most Busy Month**")
            busy_month = helper.month_activity_map(selected_user, df)
            fig = px.bar(x=busy_month.index, y=busy_month.values, labels={'x': 'Month', 'y': 'Messages'}, title='Busy Month')
            st.plotly_chart(fig)

        st.subheader("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        if selected_user == 'Overall':
            st.subheader("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(x=x.index, y=x.values, labels={'x': 'User', 'y': 'Messages'}, title='Most Busy Users')
                st.plotly_chart(fig)
            with col2:
                st.dataframe(new_df)

        st.subheader("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        st.image(df_wc.to_array())

        st.subheader("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig = px.bar(most_common_df, x=1, y=0, orientation='h', labels={"0": "Words", "1": "Count"}, title="Most Common Words")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig)
        

        #============================ textual insights of normal analysis =================
        # # === TEXTUAL SUMMARY BASED ON PLOTTED INSIGHTS ===
        st.subheader("🧠 Smart Textual Summary")

        stats = {
            "total_messages": num_messages,
            "total_words": words,
            "media_shared": num_media_messages,
            "links_shared": num_links,
            "most_common_words": most_common_df.head(10).values.tolist(),
            "most_busy_day": helper.week_activity_map(selected_user, df).idxmax(),
            "most_busy_month": helper.month_activity_map(selected_user, df).idxmax(),
        }

        if selected_user == 'Overall':
            busy_users, _ = helper.most_busy_users(df)
            stats["most_busy_users"] = busy_users.head(5).to_dict()

        # Initialize Cohere Client
        
        
        with st.spinner("Generating enhanced textual summary..."):
            st.success(textual_analysis(stats))  # Use message.content[0].text to get the generated text
            


# ====================================== CHAT SUMMARIZATION =============================================================
    if st.session_state.show_summarization:
        st.title("Chat Summarization")
        content = chatbot.summarize(df)
        st.write(content)

# ====================================== SENTIMENT ANALYSIS =============================================================
    from deep_translator import GoogleTranslator

    # Assuming: sentiment_helper and sentiment_preprocessor are already imported
    if st.session_state.show_sentiment:
        st.title("Sentiment Analysis")

        # Decode uploaded file
        bytes_data = uploaded_file.getvalue()
        df = bytes_data.decode("utf-8")
        data = sentiment_preprocessor.Preprocess(df)

        translator = GoogleTranslator(source='hi', target='en')
        sentiments = SentimentIntensityAnalyzer()
        
        # =========================================
        def safe_translate(msg):
            try:
                if isinstance(msg, str) and msg.strip():
                    result = translator.translate(msg)
                    print(result)
                    return result
                else:
                    return msg       
            except Exception:
                return msg

        # Translate Hinglish to English
        data["translated"] = data["message"].apply(safe_translate)
        # =========================================

        # Apply VADER on translated messages
        data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["translated"]]
        data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["translated"]]
        data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["translated"]]

        # Determine overall sentiment value
        def sentiment(d):
            if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
                return 1
            if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
                return -1
            return 0

        data['value'] = data.apply(lambda row: sentiment(row), axis=1)

        user_list = data['user'].unique().tolist()
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Show analysis wrt", user_list, key="sentiment_analysis_user_selector")

        if st.sidebar.button("Show Analysis"):
            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                st.markdown(f"### Monthly Activity Map ({label})")
                busy_month = sentiment_helper.month_activity_map(selected_user, data, sentiment_value)
                # Fix: Create DataFrame from Series
                df_month = pd.DataFrame({'month': busy_month.index, 'messages': busy_month.values})
                fig = px.bar(df_month, x='month', y='messages',
                                labels={'month': 'Month', 'messages': 'Messages'},
                                title=f"Monthly Activity - {label}", color_discrete_sequence=[color])
                st.plotly_chart(fig)

            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                st.markdown(f"### Daily Activity Map ({label})")
                busy_day = sentiment_helper.week_activity_map(selected_user, data, sentiment_value)
                # Fix: Create DataFrame from Series
                df_day = pd.DataFrame({'day': busy_day.index, 'messages': busy_day.values})
                fig = px.bar(df_day, x='day', y='messages',
                                labels={'day': 'Day', 'messages': 'Messages'},
                                title=f"Daily Activity - {label}", color_discrete_sequence=[color])
                st.plotly_chart(fig)

            for label, sentiment_value in zip(["Positive", "Neutral", "Negative"], [1, 0, -1]):
                try:
                    st.markdown(f"### Weekly Activity Heatmap ({label})")
                    user_heatmap = sentiment_helper.activity_heatmap(selected_user, data, sentiment_value)
                    fig = go.Figure(data=go.Heatmap(
                        z=user_heatmap.values,
                        x=user_heatmap.columns,
                        y=user_heatmap.index,
                        colorscale='Viridis'))
                    fig.update_layout(title=f"Weekly Heatmap - {label}")
                    st.plotly_chart(fig)
                except:
                    st.image('error.webp')

            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                st.markdown(f"### Daily Timeline ({label})")
                daily_timeline = sentiment_helper.daily_timeline(selected_user, data, sentiment_value)
                fig = px.line(daily_timeline, x='only_date', y='message', title=f"Daily Timeline - {label}", markers=True, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                st.markdown(f"### Monthly Timeline ({label})")
                timeline = sentiment_helper.monthly_timeline(selected_user, data, sentiment_value)
                fig = px.line(timeline, x='time', y='message', title=f"Monthly Timeline - {label}", markers=True, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            if selected_user == 'Overall':
                for label, sentiment_value in zip(["Positive", "Neutral", "Negative"], [1, 0, -1]):
                    st.markdown(f"### Most {label} Contribution")
                    df_sent = sentiment_helper.percentage(data, sentiment_value)
                    st.dataframe(df_sent)

                for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                    st.markdown(f"### Most {label} Users")
                    x = data['user'][data['value'] == sentiment_value].value_counts().head(10)
                    # Fix: Create DataFrame from Series
                    df_users = pd.DataFrame({'user': x.index, 'count': x.values})
                    fig = px.bar(df_users, x='user', y='count', 
                            labels={'user': 'User', 'count': 'Messages'}, 
                            title=f"Top {label} Users", color_discrete_sequence=[color])
                    st.plotly_chart(fig)

            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                try:
                    st.markdown(f"### {label} Word Cloud")
                    df_wc = sentiment_helper.create_wordcloud(selected_user, data, sentiment_value)
                    st.image(df_wc.to_array())
                except:
                    st.image('error.webp')

            for label, sentiment_value, color in zip(["Positive", "Neutral", "Negative"], [1, 0, -1], ['green', 'grey', 'red']):
                try:
                    most_common_df = sentiment_helper.most_common_words(selected_user, data, sentiment_value)
                    st.markdown(f"### {label} Words")
                    fig = px.bar(most_common_df, x=1, y=0, orientation='h', 
                            labels={"0": "Words", "1": "Count"}, 
                            title=f"Most Common {label} Words", color_discrete_sequence=[color])
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)
                except:
                    st.image('error.webp')
