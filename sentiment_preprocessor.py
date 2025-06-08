
import re
import pandas as pd

# To convert text into data frame in desired form
def Preprocess(data):
    # Regular expression
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    
    # Split text file into messages & dates based on pattern
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    # Creating data frame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
    # convert dates type
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ', errors='coerce')
    except:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])
    
    # Creating new columns
    df['user'] = users
    df['message'] = messages
    
    # Remove columns of no use
    df.drop(columns=['user_message'], inplace=True)
    
    # Extract date
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Remove entries having user as group_notification
    df = df[df['user'] != 'group_notification']
    
    return df
