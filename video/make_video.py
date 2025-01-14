# from pytube import YouTube
# import os
# import urllib.request

# # Custom User-Agent 설정
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
# opener = urllib.request.build_opener()
# opener.addheaders = [(key, value) for key, value in headers.items()]
# urllib.request.install_opener(opener)

# def downloadYouTube(videourl, path):

#     yt = YouTube(videourl)
#     yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
#     if not os.path.exists(path):
#         os.makedirs(path)
#     yt.download(path)

# # video_path = 'https://youtu.be/cO0yKl_xXBQ'
# video_path = 'https://www.youtube.com/watch?v=cO0yKl_xXBQ'
# downloadYouTube(video_path, './video')


import yt_dlp

url = "https://www.youtube.com/watch?v=cO0yKl_xXBQ"

# 다운로드 옵션 설정
ydl_opts = {
    "outtmpl": "./video/%(title)s.%(ext)s",  # 파일 저장 위치와 이름
    "format": "best",  # 최고 품질 선택
    "quiet": False,  # 로그 출력
    "nocheckcertificate": True,  # 인증서 체크 무시
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])