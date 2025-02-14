import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, cm
import numpy as np

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    #st.dataframe(df)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        st.write("") #to increase spacing figures
        st.write("")
        st.write("")

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")

        #weekly activity map
        st.title("Weekly Activity Map")
        #user_heatmap = helper.activity_heatmap(selected_user,df)
        #fig,ax = plt.subplots()
        #ax = sns.heatmap(user_heatmap)
        #st.pyplot(fig)
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()

        sns.heatmap(user_heatmap, cmap='viridis', linewidths=0.001, linecolor='white',
                    cbar_kws={'label': 'Activity Level'}, ax=ax)
        plt.xlabel('Hour of the Day', fontsize=10)
        plt.ylabel('Day of the Week', fontsize=10)
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                #ax.bar(x.index, x.values,color='red')
                #plt.xticks(rotation='vertical')
                #st.pyplot(fig)
                colors = cm.viridis(np.linspace(0, 1, len(x.index)))  # Create a colormap
                ax.bar(x.index, x.values, color=colors)  # Use the colormap for the bars
                ax.set_xlabel('User')  # Add x-axis label
                ax.set_ylabel('Activity Count')  # Add y-axis label
                ax.set_xticklabels(x.index, rotation="vertical")
                ax.grid(True)  # Add grid lines
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        st.write("")
        st.write("")
        st.write("")

        # WordCloud
        st.title("Wordcloud")
        rcParams['font.family'] = 'Segoe UI Emoji'
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most common words')
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            rcParams['font.family'] = 'Segoe UI Emoji'
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Count'].head(),labels=emoji_df['Emoji'].head(),autopct="%0.2f")
            st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")


        # Sentiment Analysis
        selected_user, df = helper.sentiment_data(selected_user, df)



        #bar graph of sentiment score
        sentiment_labels, sentiment_scores = helper.sentiment_score(selected_user, df)
        st.title('Sentiment Score of WhatsApp Messages')
        fig, ax = plt.subplots()  # Create a figure and axis object
        ax.bar(sentiment_labels, sentiment_scores, color=['green', 'red', 'blue'])
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Score')
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")


        # pie chart
        sentiment_labels, sentiment_counts = helper.sentiment_score_percent(selected_user, df)
        st.title('Percentile Sentiment Analysis')
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', startangle=90,
                colors=['green', 'red', 'blue'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")


        # sentiment distribution
        st.title('Sentiment Distribution in WhatsApp Messages')

        # Create a figure with 3 subplots (1 row, 3 columns)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot for Positive sentiment
        axs[0].hist(df["positive"], bins=20, alpha=0.7, color='green')
        axs[0].set_title('Positive Sentiment Distribution')
        axs[0].set_xlabel('Sentiment Score')
        axs[0].set_ylabel('Frequency')

        # Plot for Negative sentiment
        axs[1].hist(df["negative"], bins=20, alpha=0.7, color='red')
        axs[1].set_title('Negative Sentiment Distribution')
        axs[1].set_xlabel('Sentiment Score')
        axs[1].set_ylabel('Frequency')

        # Plot for Neutral sentiment
        axs[2].hist(df["neutral"], bins=20, alpha=0.7, color='blue')
        axs[2].set_title('Neutral Sentiment Distribution')
        axs[2].set_xlabel('Sentiment Score')
        axs[2].set_ylabel('Frequency')

        # Pass the figure to st.pyplot() to display the plot in Streamlit
        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")




        # word cloud
        positive_wordcloud, negative_wordcloud = helper.sentiment_wordcloud(selected_user, df)
        col1, col2 =st.columns(2)
        # Create a figure and axis objects
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display Positive Word Cloud
        with col1:
         st.title('Positive Messages Word Cloud')
         axes[0].imshow(positive_wordcloud, interpolation='bilinear')
         axes[0].axis('off')  # Hide axes

        # Display Negative Word Cloud
        with col2:
         st.title('Negative Messages Word Cloud')
         axes[1].imshow(negative_wordcloud, interpolation='bilinear')
         axes[1].axis('off')  # Hide axes

        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")

        # sentiment correlation
        correlation_matrix = helper.sentiment_relation(selected_user, df)
        st.title('Sentiment Score Correlation')
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

        st.pyplot(fig)

        st.write("")
        st.write("")
        st.write("")


        # sentiment trend over time
        sentiment_daily = helper.sentiment_trend_over_time(selected_user, df)
        st.title('Sentiment Trend Over Time')

        fig, ax = plt.subplots(figsize=(12, 6))  # Create the figure and axes with the desired size

        # Plot the sentiment data
        ax.plot(sentiment_daily.index, sentiment_daily['positive'], label='Positive', color='green')
        ax.plot(sentiment_daily.index, sentiment_daily['negative'], label='Negative', color='red')
        ax.plot(sentiment_daily.index, sentiment_daily['neutral'], label='Neutral', color='blue')

        # Set the labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')


        # Show the legend
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)














