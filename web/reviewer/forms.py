from django import forms
class ReviewActionForm(forms.Form):
    crop_filename = forms.CharField(widget=forms.HiddenInput())
    predicted = forms.CharField(required=False, widget=forms.HiddenInput())
    distance_value = forms.FloatField(required=False, widget=forms.HiddenInput())
    action = forms.ChoiceField(choices=[("approve","Approve"),("unknown","Unknown"),("override","Override")])
    override_label = forms.CharField(required=False, max_length=128)
