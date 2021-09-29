from django import forms
class InputForm(forms.Form):
   URL = forms.CharField(max_length = 100)
   FEVER =  forms.CharField(max_length = 1)
   OC =  forms.CharField(max_length = 1)
   health_status = forms.CharField(max_length = 20)
