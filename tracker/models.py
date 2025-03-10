from django.conf import settings
from django.db import models

class Video(models.Model):
    uploaded_video = models.FileField(upload_to='uploads/')
    processed_video = models.FileField(upload_to='processed_videos/', null=True, blank=True)
    pushup_count = models.IntegerField(default=0)

    def __str__(self):
        return f"Video {self.id} - Push-ups: {self.pushup_count}"

    def get_processed_video_url(self):
        """
        Returns the correct media URL for the processed video in the format: /media/uploads/{video_name}
        """
        if self.processed_video:
            video_filename = self.processed_video.name.split('/')[-1]  # Extract only the file name
            return f"{settings.MEDIA_URL}uploads/{video_filename}"
        return None
