@echo off
echo Starting Django Fake News Detection App...

echo Checking Django project...
python manage.py check
if errorlevel 1 (
    echo Django check failed.
    pause
    exit /b 1
)

echo Applying database migrations...
python manage.py migrate
if errorlevel 1 (
    echo Migration failed.
    pause
    exit /b 1
)

echo Starting development server...
python manage.py runserver

pause
