from django.urls import path

from nlp.views import NLPFormView

app_name = 'nlp'

urlpatterns = [
    path('skipgram/', NLPFormView.as_view(), name='skipgram'),
    # path('success/', SuccessView.as_view(), name='success'),
]