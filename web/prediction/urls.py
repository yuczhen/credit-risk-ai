from django.urls import path

from . import views

app_name = "prediction"

urlpatterns = [
    path("", views.prediction_page, name="prediction_page"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_predict, name="upload_predict"),
    path("download/csv/", views.download_csv, name="download_csv"),
    path("download/excel/", views.download_excel, name="download_excel"),
    path("download/batch/", views.download_batch_result, name="download_batch_result"),
    path("download/template/", views.download_template, name="download_template"),
]
