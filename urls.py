from django.urls import path
from . import views
urlpatterns = [
    path('',views.home,name='home'),
    path('detect/',views.detect,name='detect'),
    path('detect_img/',views.detect_img,name='detect_img'),
    path('surveillance/',views.surveillance,name = 'surveillance'),
    path('registercase/',views.registercase,name='registercase'),
    path('missing/',views.missing,name='missing'),
    path('location/',views.location,name='location'),
    path('delete/<int:person_id>/',views.delete_person, name='delete_person'),
    path('update/<int:person_id>/',views.update_person, name='update_person'),
    path('login/',views.login,name='login'),
    path('register/',views.register,name='register'),
    path('logout/',views.logout,name='logout'),
]