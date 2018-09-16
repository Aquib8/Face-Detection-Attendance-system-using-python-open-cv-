# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .models import *
from django.shortcuts import render, redirect
from django.http import HttpResponse
# Create your views here.
def index(request):
    records = Records.objects.all()[:10]    #getting the first 10 records
    context = {
        'records': records
    }
    return render(request, 'records.html', context)

# def details(request, id):
#     students = Students.objects.get(id=id)
#     context = {
#         'students' : students
#     }
#     return render(request, 'studentInfo.html', context)

