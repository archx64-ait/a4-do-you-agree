from django import forms

style = forms.Select(attrs={"class": "form-control"})


class NLPForm(forms.Form):
    premise = forms.CharField(required=True, widget=forms.Textarea(attrs={"rows":"5"}))
    hypothesis = forms.CharField(required=True, widget=forms.Textarea(attrs={"rows":"5"}))
