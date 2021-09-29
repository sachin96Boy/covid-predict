from django.urls import path
from sound_prediction import views
from django.views.generic import TemplateView
from django.conf.urls import url

urlpatterns = [
    path('', views.homepage, name='homepage'),
    # path('submit',views.give_prediction,name ='give_prediction' ),
    # path('connection/',TemplateView.as_view(template_name = 'homepage.html')),
    path('login/',views.login, name = 'login')
]


