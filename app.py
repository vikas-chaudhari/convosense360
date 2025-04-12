import nltk
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import chatbot
import re
import sentiment_preprocessor,sentiment_helper
import pandas as pd
import numpy as np
# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.download('vader_lexicon')
st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    # Initialize session state for all three views
    if "show_normal_analysis" not in st.session_state:
        st.session_state.show_normal_analysis = False
    if "show_summarization" not in st.session_state:
        st.session_state.show_summarization = False
    if "show_sentiment" not in st.session_state:
        st.session_state.show_sentiment = False
    # Buttons to toggle views
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
    # User selection
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    if st.session_state.show_normal_analysis:
        selected_user = st.sidebar.selectbox("Show analysis wrt", user_list, key="normal_analysis_user_selector")
    # ========== NORMAL ANALYSIS ==========
    if st.session_state.show_normal_analysis:
        st.title("Normal Chat Analysis")
        # Stats Area
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
        # Monthly timeline
        st.subheader("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        # Daily timeline
        st.subheader("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        # Activity map
        st.subheader("Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most Busy Day**")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("**Most Busy Month**")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        st.subheader("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)
        if selected_user == 'Overall':
            st.subheader("Most Busy Users")
            x, new_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        st.subheader("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
        st.subheader("Most Common Words")
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
    # ========== SUMMARIZATION ================================================================================
    if st.session_state.show_summarization:
        st.title("Chat Summarization")
        content = chatbot.summarize(df)
        st.write(content)
    # ========== SENTIMENT ANALYSIS ===========================================================================
    if st.session_state.show_sentiment:
        st.title("Sentiment Analysis")
        # Getting byte form & then decoding
        bytes_data = uploaded_file.getvalue()
        df = bytes_data.decode("utf-8")
        # Perform preprocessing
        data = sentiment_preprocessor.Preprocess(df)
        # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        # Object
        sentiments = SentimentIntensityAnalyzer()
        # Creating different columns for (Positive/Negative/Neutral)
        data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]] # Positive
        data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]] # Negative
        data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]] # Neutral
        # To indentify true sentiment per row in message column
        def sentiment(d):
            if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
                return 1
            if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
                return -1
            if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
                return 0
        # Creating new column & Applying function
        data['value'] = data.apply(lambda row: sentiment(row), axis=1)
        # User names list
        user_list = data['user'].unique().tolist()
        # Sorting
        user_list.sort()
        # Insert "Overall" at index 0
        user_list.insert(0, "Overall")
        if st.session_state.show_sentiment:
            selected_user = st.sidebar.selectbox("Show analysis wrt", user_list, key="sentiment_analysis_user_selector")
        if st.sidebar.button("Show Analysis"):
            # Monthly activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)
                busy_month = sentiment_helper.month_activity_map(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)
                busy_month = sentiment_helper.month_activity_map(selected_user, data, 0)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)
                busy_month = sentiment_helper.month_activity_map(selected_user, data, -1)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            # Daily activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)
                busy_day = sentiment_helper.week_activity_map(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)
                busy_day = sentiment_helper.week_activity_map(selected_user, data, 0)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)
                busy_day = sentiment_helper.week_activity_map(selected_user, data, -1)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            # Weekly activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)
                    user_heatmap = sentiment_helper.activity_heatmap(selected_user, data, 1)
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')
            with col2:
                try:
                    st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)
                    user_heatmap = sentiment_helper.activity_heatmap(selected_user, data, 0)
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')
            with col3:
                try:
                    st.markdown("<h3 style='text-align: center; color: white;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)
                    
                    user_heatmap = sentiment_helper.activity_heatmap(selected_user, data, -1)
                    
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')
            # Daily timeline
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline(Positive)</h3>",unsafe_allow_html=True)
                daily_timeline = sentiment_helper.daily_timeline(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline(Neutral)</h3>",unsafe_allow_html=True)
                daily_timeline = sentiment_helper.daily_timeline(selected_user, data, 0)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Daily Timeline(Negative)</h3>",unsafe_allow_html=True)
                daily_timeline = sentiment_helper.daily_timeline(selected_user, data, -1)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            # Monthly timeline
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline(Positive)</h3>",unsafe_allow_html=True)
                timeline = sentiment_helper.monthly_timeline(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline(Neutral)</h3>",unsafe_allow_html=True)
                timeline = sentiment_helper.monthly_timeline(selected_user, data,0)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: white;'>Monthly Timeline(Negative)</h3>",unsafe_allow_html=True)
                timeline = sentiment_helper.monthly_timeline(selected_user, data,-1)              
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            # Percentage contributed
            if selected_user == 'Overall':
                col1,col2,col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                    x = sentiment_helper.percentage(data, 1)
                    # Displaying
                    st.dataframe(x)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                    y = sentiment_helper.percentage(data, 0)
                    # Displaying
                    st.dataframe(y)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                    z = sentiment_helper.percentage(data, -1)
                    
                    # Displaying
                    st.dataframe(z)
            # Most Positive,Negative,Neutral User...
            if selected_user == 'Overall':
                # Getting names per sentiment
                x = data['user'][data['value'] == 1].value_counts().head(10)
                y = data['user'][data['value'] == -1].value_counts().head(10)
                z = data['user'][data['value'] == 0].value_counts().head(10)
                col1,col2,col3 = st.columns(3)
                with col1:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Users</h3>",unsafe_allow_html=True)
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(z.index, z.values, color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Users</h3>",unsafe_allow_html=True)
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(y.index, y.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
            # WORDCLOUD......
            col1,col2,col3 = st.columns(3)
            with col1:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Positive WordCloud</h3>",unsafe_allow_html=True)
                    # Creating wordcloud of positive words
                    df_wc = sentiment_helper.create_wordcloud(selected_user, data,1)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')
            with col2:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Neutral WordCloud</h3>",unsafe_allow_html=True)
                    # Creating wordcloud of neutral words
                    df_wc = sentiment_helper.create_wordcloud(selected_user, data,0)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')
            with col3:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Negative WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of negative words
                    df_wc = sentiment_helper.create_wordcloud(selected_user, data,-1)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')
            # Most common positive words
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    # Data frame of most common positive words.
                    most_common_df = sentiment_helper.most_common_words(selected_user, data,1)
                    
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Positive Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1],color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')
            with col2:
                try:
                    # Data frame of most common neutral words.
                    most_common_df = sentiment_helper.most_common_words(selected_user, data,0)
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Neutral Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1],color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')
            with col3:
                try:
                    # Data frame of most common negative words.
                    most_common_df = sentiment_helper.most_common_words(selected_user, data,-1)
                    # heading
                    st.markdown("<h3 style='text-align: center; color: white;'>Negative Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1], color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')
# ================================================================================================================