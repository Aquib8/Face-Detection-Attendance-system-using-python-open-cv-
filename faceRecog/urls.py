"""faceRecog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.urls import path

from faceRecog import views as app_views

urlpatterns = [
    url(r'^$', app_views.index),

    url(r'^studentlogout', app_views.studentlogout),
    url(r'^teacherlogout', app_views.teacherlogout),
    url(r'^adminlogout', app_views.adminlogout),

    url(r'^adminmain/', app_views.adminmain),
    url(r'^students/', app_views.student),
    url(r'^teachers/', app_views.teacher),
    url(r'^error_image$', app_views.errorImg),
    url(r'^create_dataset$', app_views.create_dataset),
    url(r'^trainer$', app_views.trainer),
    #url(r'^eigen_train$', app_views.eigenTrain),
    url(r'^detect/(?P<id>\w{0,50})/$', app_views.detect),
    #url(r'^detect_image$', app_views.detectImage),
    url(r'^admin/', admin.site.urls),
    url(r'^records/', include('records.urls')),
    url(r'^adminstudent/', app_views.adminstudent),
    url(r'^adminteacher/', app_views.adminteacher),
    url(r'^admindepartment/', app_views.admindepartment),
    url(r'^adminroom/', app_views.adminroom),
    url(r'^admincourses/', app_views.admincourses),
    url(r'^adminpost/', app_views.adminpost),
    url(r'^adminsemester/', app_views.adminsemester),
    url(r'^admingrade/', app_views.admingrade),
    url(r'^adminclassinfo/', app_views.adminclassinfo),
    url(r'^admintiming/', app_views.admintiming),
    url(r'^teacheraddcourse/', app_views.teacheraddcourse),
    url('studentaddcourse/', app_views.studentaddcourse),
    # url(r'^studentaddcourse/(?P<id>\d)/$', include('records.urls')),
    url(r'^teacheraddcourse_two/', app_views.teacheraddcourse_two),
    url(r'^studentlist/', app_views.studentlist),
    url(r'^teacherlist/', app_views.teacherlist),
    url(r'^takeattandance/', app_views.takeattandance),
    # url(r'^edit_product/(?P<id>\w{0,50})/$', views.edit_product),
    url(r'^coursestudentlist/(?P<id>\w{0,50})/$', app_views.coursestudentlist),
    url(r'^attandancelist/', app_views.attandancelist),
    url(r'submitattandance/(?P<id>\w{0,50})/$', app_views.submitattandance),
    url(r'enrolledcourses/', app_views.enrolledcourses),

]

