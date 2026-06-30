from django.db import models

# Create your models here.
class Document(models.Model):
    document_id = models.CharField(max_length=64, unique=True)  # SHA-256 hash
    filename = models.CharField(max_length=255)
    mimetype = models.CharField(max_length=50)
    size = models.PositiveIntegerField()
    hash = models.CharField(max_length=64, unique=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)