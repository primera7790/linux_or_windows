import os
from pathlib import Path

from pyyoutube import Api
from dotenv import load_dotenv

load_dotenv(Path(Path(__file__).parent.parent, 'env_data/.env'))

YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
api = Api(api_key=YOUTUBE_API_KEY)

print(api)


