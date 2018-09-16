# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from datetime import datetime
from django.db import models
from django.utils import timezone


# Create your models here.

# 1           STUDENTS

class Students(models.Model):
    institute_id = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    confirm_password = models.CharField(max_length=50)
    name = models.CharField(max_length=50)
    address = models.CharField(max_length=50)
    phone_number = models.CharField(max_length=50)
    email_address = models.CharField(max_length=50)
    role_id = models.ForeignKey('Roles', on_delete=models.CASCADE)
    pic_status = models.BooleanField(default=False)
    dept_id = models.ForeignKey('Department', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Students"


# 2               Teachers

class Teachers(models.Model):
    initial = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    dept_id = models.ForeignKey('Department', on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    post_id = models.ForeignKey('Post', on_delete=models.CASCADE)
    role_id = models.ForeignKey('Roles', on_delete=models.CASCADE)
    office_location = models.CharField(max_length=45)


    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Teachers"


# 3                   ADMIN

class Admins(models.Model):
    username = models.CharField(max_length=50)
    name = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    role_id = models.ForeignKey('Roles', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Admins"


# 4                   COURSE

class Course(models.Model):
    name = models.CharField(max_length=50)
    title = models.CharField(max_length=50)
    description = models.CharField(max_length=50)
    credit = models.CharField(max_length=50)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Course"


# 5                   POST

class Post(models.Model):
    name = models.CharField(max_length=50)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Post"


# 6                  Roles

class Roles(models.Model):
    weight = models.CharField(max_length=100)
    name = models.CharField(max_length=50)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Roles"


# 7                   Teacher_courses

class Teacher_courses(models.Model):
    course_id = models.ForeignKey('Course', on_delete=models.CASCADE)
    teacher_id = models.ForeignKey('Teachers', on_delete=models.CASCADE)
    status = models.BooleanField(default=False)
    class_info_id = models.ForeignKey('Class_info', on_delete=models.CASCADE)
    class_count = models.IntegerField(default=0)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Teacher_courses"


# 8                Class_info

class Class_info(models.Model):
    timing_id = models.ForeignKey('Timing', on_delete=models.CASCADE)
    semester_id = models.ForeignKey('Semester', on_delete=models.CASCADE)
    rooms_id = models.ForeignKey('Rooms', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Class_info"


# 9              Rooms

class Rooms(models.Model):
    building = models.CharField(max_length=40)
    room_no = models.CharField(max_length=50)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Rooms"


# 10         Grades

class Grades(models.Model):
    point = models.DecimalField(max_digits=3, decimal_places=2)
    name = models.CharField(max_length=40)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Grades"


# 11         Attendance

class Attendance(models.Model):
    student_course_id = models.ForeignKey('Student_courses', on_delete=models.CASCADE)
    status = models.BooleanField(default=True)
    created = models.DateTimeField(editable=False)
    modified = models.DateTimeField(null=True)

    def save(self, *args, **kwargs):
        if not self.id:
            self.created = timezone.now()
        self.modified = timezone.now()
        return super(Attendance, self).save(*args, **kwargs)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Attendance"


# 12         Student_courses

class Student_courses(models.Model):
    student_id = models.ForeignKey('Students', on_delete=models.CASCADE)
    teacher_course_id = models.ForeignKey('Teacher_courses', on_delete=models.CASCADE)
    grade_id = models.ForeignKey('Grades', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Student_courses"


# 13         Department

class Department(models.Model):
    name = models.CharField(max_length=50)
    location = models.CharField(max_length=50)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Depertment"


# 14         Semester

class Semester(models.Model):
    name = models.CharField(max_length=50)
    start_date = models.DateField()
    end_date = models.DateField()
    year = models.DateField()

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Semester"


# 15  Timing

class Timing(models.Model):
    days = models.CharField(max_length=50)
    start_time = models.TimeField()
    end_time = models.TimeField()

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = "Timing"


class Records(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50, null=True)
    residence = models.CharField(max_length=50, null=True)
    country = models.CharField(max_length=50, null=True)
    education = models.CharField(max_length=150, null=True)
    occupation = models.CharField(max_length=150, null=True)
    marital_status = models.CharField(max_length=50, null=True)
    bio = models.TextField()
    recorded_at = models.DateTimeField(default=datetime.now, blank=True)

    def __str__(self):
        return self.first_name

    class Meta:
        verbose_name_plural = "Records"
