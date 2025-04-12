import re
import pandas as pd

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    # convert message_date type
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    return df


# import re
# import pandas as pd

# def preprocess(data):
#     # Adjusted regex to match the provided chat format
#     pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(:\d{2})?\s(?:AM|PM|am|pm)\]\s'
    
#     # Split messages and extract dates
#     messages = re.split(pattern, data)[1:]  # Messages are after the timestamps
#     dates = re.findall(pattern, data)      # Extract all timestamps

#     # Create a DataFrame with extracted messages and dates
#     df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
#     # Convert message_date into a proper datetime object
#     df['message_date'] = pd.to_datetime(
#         df['message_date'].str.strip('[]'), 
#         format='%d/%m/%y, %I:%M:%S %p',  # For seconds
#         errors='coerce'  # Handle potential mismatches
#     )
    
#     # Fallback for timestamps without seconds
#     df['message_date'] = df['message_date'].fillna(
#         pd.to_datetime(
#             df['message_date'].str.strip('[]'), 
#             format='%d/%m/%y, %I:%M %p',
#             errors='coerce'
#         )
#     )
    
#     # Rename column for clarity
#     df.rename(columns={'message_date': 'date'}, inplace=True)

#     # Separate users and messages
#     users = []
#     messages = []
#     for message in df['user_message']:
#         entry = re.split(r'([\w\W]+?):\s', message)
#         if entry[1:]:  # If the message has a user name
#             users.append(entry[1])
#             messages.append(" ".join(entry[2:]))
#         else:
#             users.append('group_notification')
#             messages.append(entry[0])
    
#     # Assign the users and messages into the DataFrame
#     df['user'] = users
#     df['message'] = messages
#     df.drop(columns=['user_message'], inplace=True)

#     # Extract additional date and time features
#     df['only_date'] = df['date'].dt.date
#     df['year'] = df['date'].dt.year
#     df['month_num'] = df['date'].dt.month
#     df['month'] = df['date'].dt.month_name()
#     df['day'] = df['date'].dt.day
#     df['day_name'] = df['date'].dt.day_name()
#     df['hour'] = df['date'].dt.hour
#     df['minute'] = df['date'].dt.minute

#     # Create period feature (hour ranges)
#     period = []
#     for hour in df[['day_name', 'hour']]['hour']:
#         if hour == 23:
#             period.append(str(hour) + "-" + str('00'))
#         elif hour == 0:
#             period.append(str('00') + "-" + str(hour + 1))
#         else:
#             period.append(str(hour) + "-" + str(hour + 1))

#     df['period'] = period
#     print(df)
#     return df
