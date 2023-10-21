from django.urls import path
from admin_volt import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    # Index
    # path('index/', views.index, name="index"),

    # Pages
    path('<message>', views.dashboard, name="dashboard"),
    path('', views.dashboard, name="dashboard"),
    path('pages/settings/', views.settings, name="settings"),

    # Authentication
    path('accounts/register/', views.register_view, name="register"),
    path('accounts/login/', views.UserLoginView.as_view(), name="login"),
    path('accounts/logout/', views.logout_view, name="logout"),
    path('accounts/password-change/', views.UserPasswordChangeView.as_view(), name='password_change'),
    path('accounts/password-change-done/', auth_views.PasswordChangeDoneView.as_view(
        template_name='accounts/password-change-done.html'
    ), name="password_change_done"),
    path('accounts/password-reset/', views.UserPasswordResetView.as_view(), name="password_reset"),
    path('accounts/password-reset-confirm/<uidb64>/<token>/',
        views.UserPasswrodResetConfirmView.as_view(), name="password_reset_confirm"
    ),
    path('accounts/password-reset-done/', auth_views.PasswordResetDoneView.as_view(
        template_name='accounts/password-reset-done.html'
    ), name='password_reset_done'),
    path('accounts/password-reset-complete/', auth_views.PasswordResetCompleteView.as_view(
        template_name='accounts/password-reset-complete.html'
    ), name='password_reset_complete'),

    # Errors
    path('error/404/', views.error_404, name="error_404"),
    path('error/500/', views.error_500, name="error_500"),

    path('get_checked/', views.get_checked, name='get_checked'),
]
