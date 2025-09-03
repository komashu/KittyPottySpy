from django.urls import path
from .views import review_list_view, review_action_view, labeled_frames_view
urlpatterns = [
    path("", review_list_view, name="review_list"),
    path("act/", review_action_view, name="review_action"),
    path("labeled/", labeled_frames_view, name="labeled_frames"),
]
