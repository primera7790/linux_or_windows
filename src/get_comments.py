import os
from pathlib import Path

from dotenv import load_dotenv
import json

import requests
from pyyoutube import Api


load_dotenv(Path(Path(__file__).parent.parent, 'config/.env'))
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')


def get_data(videoId, maxResults, nextPageToken):
    """
    Получение информации со страницы с видео
    """
    YOUTUBE_URI = ('https://www.googleapis.com/youtube/v3/commentThreads?key={KEY}&textFormat=plainText&'
                   'part=snippet&videoId={videoId}&maxResults={maxResults}&pageToken={nextPageToken}')
    format_youtube_uri = YOUTUBE_URI.format(KEY=YOUTUBE_API_KEY,
                                            videoId=videoId,
                                            maxResults=maxResults,
                                            nextPageToken=nextPageToken)
    content = requests.get(format_youtube_uri).text
    data = json.loads(content)
    return data


def get_text_of_comment(data):
    """
    Получение текста комментария
    """
    comments = set()
    for item in data['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.add(comment)
    return comments


def get_all_comments(query, count_video, limit, maxResults, nextPageToken):
    """
    Получение указанного числа комментариев под видео
    """
    api = Api(api_key=YOUTUBE_API_KEY)
    video_by_keywords = api.search_by_keywords(q=query,
                                               search_type=['video'],
                                               count=count_video,
                                               limit=limit)
    video_id = [x.id.videoId for x in video_by_keywords.items]

    comments_all = []
    for id in video_id:
        try:
            data = get_data(id,
                            maxResults=maxResults,
                            nextPageToken=nextPageToken)
            comment = list(get_text_of_comment(data))
            comments_all.append(comment)
        except:
            continue
    comments = sum(comments_all, [])
    return comments
