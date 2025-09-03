from django.db import models
class Feedback(models.Model):
    DECISION_CHOICES = [
        ("approve", "Approve"),
        ("unknown", "Unknown"),
        ("override", "Override"),
    ]
    created_at_epoch = models.IntegerField()
    crop_filename = models.CharField(max_length=512)
    predicted_label = models.CharField(max_length=128, blank=True, null=True)
    distance_value = models.FloatField(blank=True, null=True)
    decision = models.CharField(max_length=16, choices=DECISION_CHOICES)
    corrected_label = models.CharField(max_length=128, blank=True, null=True)
    class Meta:
        ordering = ["-id"]
