# vizapi
A comprehensive Modal FastAPI script for creating music visualizers quickly, in the cloud.

Usage:
```
curl -fsS "$API/jobs" \
  -H "X-API-Key: $KEY" \
  -F "audio=@audio.wav" \
  -F "cover=@artwork.png" \
  -F "title=example title" \
  -F "artist=example artist" \
  -F "resolution=1080p" \
  -F "fps=60" \
  -F "codec=av1" \
  -F "orientation=landscape"
```
```
curl -fsS "$API/jobs/<your_job_id_goes_here>" \
  -H "X-API-Key: $KEY"
```
```
curl -fL "$API/jobs/<the_same_job_id_goes_here>/video" \
  -H "X-API-Key: $KEY" \
  -o video_file_name.mp4
```