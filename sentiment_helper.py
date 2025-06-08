import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Function to calculate weekly activity map
def week_activity_map(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['day_name'].value_counts()

# Function to calculate monthly activity map
def month_activity_map(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['month'].value_counts()

# Function to create activity heatmap
def activity_heatmap(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    
    user_heatmap = df.pivot_table(index='day_name', columns='hour', values='message', aggfunc='count').fillna(0)
    return user_heatmap

# Function for daily timeline
def daily_timeline(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline

# Function for monthly timeline
def monthly_timeline(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline

# Function to calculate message contributions in terms of percentage
def percentage(df, k):
    df = round((df['user'][df['value'] == k].value_counts() / df[df['value'] == k].shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df

# Function to generate a word cloud
def create_wordcloud(selected_user, df, k):
    stop_words = set(open('stop_hinglish.txt', 'r').read().splitlines())
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    def remove_stop_words(message):
        return " ".join([word for word in message.lower().split() if word not in stop_words])
    
    temp['message'] = temp['message'].apply(remove_stop_words)
    temp = temp[temp['value'] == k]
    
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(" ".join(temp['message']))
    return wc

# Function to calculate most common words
def most_common_words(selected_user, df, k):
    stop_words = set(open('stop_hinglish.txt', 'r').read().splitlines())
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    words = []
    for message in temp['message'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
                
    common_words = pd.DataFrame(Counter(words).most_common(20))
    return common_words

# Function to get the top users based on sentiment value
def top_users(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    
    user_counts = df['user'].value_counts().head(10)
    user_counts_df = user_counts.reset_index()
    user_counts_df.columns = ['user', 'count']
    return user_counts_df

# Function to calculate the sentiment distribution of messages by hour
def sentiment_by_hour(df):
    sentiment_by_hour = df.groupby(['hour', 'value']).size().unstack().fillna(0)
    sentiment_by_hour.columns = ['Negative', 'Neutral', 'Positive']
    return sentiment_by_hour

# Function to create a pie chart of sentiment distribution
def sentiment_distribution(df):
    sentiment_count = df['value'].value_counts()
    sentiment_pie = px.pie(names=['Negative', 'Neutral', 'Positive'], values=sentiment_count, title="Sentiment Distribution")
    return sentiment_pie
