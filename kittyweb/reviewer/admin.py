from django.contrib import admin
from .models import Feedback
@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("id", "crop_filename", "predicted_label", "distance_value", "decision", "corrected_label", "created_at_epoch")
    search_fields = ("crop_filename", "predicted_label", "corrected_label")
    list_filter = ("decision",)
