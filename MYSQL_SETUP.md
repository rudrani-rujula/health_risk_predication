# MySQL setup

The Flask app uses MySQL database `hospital` by default. It reads these environment variables:

- `MYSQL_HOST` (default: `localhost`)
- `MYSQL_PORT` (default: `3306`)
- `MYSQL_DATABASE` (default: `hospital`)
- `MYSQL_USER` (default: `root`)
- `MYSQL_PASSWORD` (required for installations that use a password)

In PowerShell, set the values before starting the app:

```powershell
$env:MYSQL_HOST = "localhost"
$env:MYSQL_PORT = "3306"
$env:MYSQL_DATABASE = "hospital"
$env:MYSQL_USER = "root"
$env:MYSQL_PASSWORD = "your-mysql-password"
py -3 app.py
```

The app creates the `hospital` database and the `users` and `assessments` tables automatically. On startup it migrates existing rows from `health_analyzer.db` using `INSERT IGNORE`, so restarting the app does not duplicate migrated users or assessments.

Install dependencies with:

```powershell
py -3 -m pip install -r requirements.txt
```
