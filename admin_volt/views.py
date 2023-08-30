import random, urllib.request

import requests

from django.shortcuts import render, redirect
from admin_volt.forms import RegistrationForm, LoginForm, UserPasswordResetForm, UserPasswordChangeForm, \
    UserSetPasswordForm
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView, PasswordResetConfirmView
from django.contrib.auth import logout
from home.models import *
from home.views import requestmlresult
from django.contrib.auth.decorators import login_required
import json
import datetime
import os
# Index
def index(request):
    return render(request, 'pages/index.html')


# Dashboard
@login_required(login_url="/accounts/login/")
def dashboard(request):
    try:
        requestmlresult(request.user)
        slides = list(Slide.objects.all())
        prediction = predictionresult.objects.filter(user_id=request.user.id)
        result = [x.result[-5] for x in prediction]
        ele = eledata.objects.filter(user_id=request.user.id)
        label = [[] for _ in range(12)]
        dayusage = [[] for _ in range(12)]
        for i in ele:
            month = i.report_time.month
            usage = sum(float(x) / 4 for x in i.daliyusage.split(",")) / 1000
            usage = float('%.2f' % usage)
            label[month - 1].append(i.report_time.day)
            dayusage[month - 1].append(usage)

        mu = [float('%.2f' % sum(i)) for i in dayusage]
        wu = float('%.2f' % sum(mu))
        tree = 0.509 * wu / 550.5
        treec = []
        
        for i in range(int(tree) + 1):
            if i + 1 < tree:
                treec.append([100, 0])
            else:
                treec.append([100 * (tree - i), 100 - 100 * (tree - i)])
        context = {
            'segment': 'dashboard',
            'ele': dayusage,
            'date': label,
            'presult': result,
            "todayusage": dayusage[-1][-1],
            "monthusage": mu,
            "wholeusage": wu,
            'standard': [float('%.2f' % (sum(i) / len(i))) for i in dayusage],
            "treec": treec,
            "slides": slides[1:],
            "fslide": slides[0],
        }
    except:
        context = {}
    return render(request, 'pages/dashboard/dashboard.html', context)


# Pages
@login_required(login_url="/accounts/login/")
def transaction(request):
    context = {
        'segment': 'transactions'
    }
    return render(request, 'pages/transactions.html', context)


@login_required(login_url="/accounts/login/")
def settings(request):
    context = {
        'segment': 'settings'
    }
    return render(request, 'pages/settings.html', context)


# Tables
@login_required(login_url="/accounts/login/")
def bs_tables(request):
    context = {
        'parent': 'tables',
        'segment': 'bs_tables',
    }
    return render(request, 'pages/tables/bootstrap-tables.html', context)


# Components
@login_required(login_url="/accounts/login/")
def buttons(request):
    context = {
        'parent': 'components',
        'segment': 'buttons',
    }
    return render(request, 'pages/components/buttons.html', context)


@login_required(login_url="/accounts/login/")
def notifications(request):
    context = {
        'parent': 'components',
        'segment': 'notifications',
    }
    return render(request, 'pages/components/notifications.html', context)


@login_required(login_url="/accounts/login/")
def forms(request):
    context = {
        'parent': 'components',
        'segment': 'forms',
    }
    return render(request, 'pages/components/forms.html', context)


@login_required(login_url="/accounts/login/")
def modals(request):
    context = {
        'parent': 'components',
        'segment': 'modals',
    }
    return render(request, 'pages/components/modals.html', context)


@login_required(login_url="/accounts/login/")
def typography(request):
    context = {
        'parent': 'components',
        'segment': 'typography',
    }
    return render(request, 'pages/components/typography.html', context)


# Authentication
def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            print("Account created successfully!")
            form.save()
            return redirect('/accounts/login/')
        else:
            print("Registration failed!")
    else:
        form = RegistrationForm()

    context = {'form': form}
    return render(request, 'accounts/login-form-20/register.html', context)


class UserLoginView(LoginView):
    form_class = LoginForm
    template_name = 'accounts/login-form-20/index.html'


class UserPasswordChangeView(PasswordChangeView):
    template_name = 'accounts/password-change.html'
    form_class = UserPasswordChangeForm


class UserPasswordResetView(PasswordResetView):
    template_name = 'accounts/forgot-password.html'
    form_class = UserPasswordResetForm


class UserPasswrodResetConfirmView(PasswordResetConfirmView):
    template_name = 'accounts/reset-password.html'
    form_class = UserSetPasswordForm


def logout_view(request):
    logout(request)
    return redirect('/accounts/login/')


def lock(request):
    return render(request, 'accounts/lock.html')


# Errors
def error_404(request):
    return render(request, 'pages/examples/404.html')


def error_500(request):
    return render(request, 'pages/examples/500.html')


# Extra
def upgrade_to_pro(request):
    return render(request, 'pages/upgrade-to-pro.html')
