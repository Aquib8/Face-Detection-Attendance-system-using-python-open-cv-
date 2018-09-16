# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from .models import *

# Register your models here.
#from .models import Students
admin.site.register(Students)

#from .models import Teachers
admin.site.register(Teachers)

#from .models import Admins
admin.site.register(Admins)

#from .models import Course
admin.site.register(Course)

#from .models import Post
admin.site.register(Post)

#from .models import Roles
admin.site.register(Roles)


admin.site.register(Teacher_courses)

admin.site.register(Class_info)

admin.site.register(Rooms)

admin.site.register(Grades)

admin.site.register(Attendance)

admin.site.register(Student_courses)
admin.site.register(Department)

admin.site.register(Semester)
admin.site.register(Timing)



admin.site.register(Records)
