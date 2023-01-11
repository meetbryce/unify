import glob
import os
import psutil
import requests
import shutil
import subprocess
import time
import webbrowser

from prompt_toolkit.shortcuts import ProgressBar

class MetabaseSetup:
    def __init__(self, unify_home: str, prompt_func):
        self.home = unify_home
        self.prompt = prompt_func
        self.jdk_dir = os.path.join(self.home, "jdk")
        os.makedirs(self.jdk_dir, exist_ok=True)
        self.metabase_dir = os.path.join(self.home, "metabase")
        os.makedirs(self.metabase_dir, exist_ok=True)

    def jdk_url(self) -> str:
        # TODO: Detect platform
        return "https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.17%2B8/OpenJDK11U-jdk_aarch64_mac_hotspot_11.0.17_8.tar.gz"

    def local_jdk_path(self) -> str:
        return os.path.join(self.jdk_dir, "OpenJDK11.tar.gz")

    def metabase_jar_url(self):
        return "https://downloads.metabase.com/v0.45.1/metabase.jar"

    def local_metabase_jar(self):
        return os.path.join(self.metabase_dir, "metabase.jar")

    def clickhouse_driver_url(self) -> str:
        return "https://github.com/enqueue/metabase-clickhouse-driver/releases/download/0.9.1/clickhouse.metabase-driver.jar"

    def metabase_script(self) -> str:
        return os.path.join(self.metabase_dir, "metabase.sh")

    def download_file(self, label:str, url: str, dest: str):
        # Streaming, so we can iterate over the response.
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        with ProgressBar(f"Downloading {label}") as pb:
            blocks = int(total_size_in_bytes / block_size)
            count = 0
            with open(dest, 'wb') as file:
                for data in pb(response.iter_content(block_size), total=blocks):
                    count += len(data)
                    file.write(data)
        if total_size_in_bytes != count:
            print(f"ERROR, download size {count} doesn't match expected size {total_size_in_bytes}")

    def ensure_java_installed(self):
        java_path = shutil.which("java")
        if java_path:
            return java_path
        else:
            choice = self.prompt("Do you want to install Java (y/n)?")
            if choice != "y":
                return False
			# Download OpenJDK MacOSX tgz
            self.download_file("JDK", self.jdk_url(), self.local_jdk_path())
			# 
			# Untar JDK into HOME/jdk
            subprocess.check_output(["tar", "xzf", self.local_jdk_path()], cwd=self.jdk_dir)
            self.actual_jdk_path = None
            for path in glob.iglob(self.jdk_dir + "/**/java"):
                if os.path.isfile(path):
                    return path
            return None

    def write_metabase_script(self):
        with open(self.metabase_script(), "w") as f:
            f.write(
f"""#/bin/env sh
cd {self.metabase_dir}
MB_PLUGINS_DIR=./plugins; java -jar ./metabase.jar
""")
        os.chmod(self.metabase_script(), 0o755)

    def ensure_metabase_installed(self):
        if not os.path.exists(self.local_metabase_jar()):
            self.download_file("Metabase", self.metabase_jar_url(), self.local_metabase_jar())
        driver_path = os.path.join(self.metabase_dir, "plugins", "clickhouse.metabase-driver.jar")
        if not os.path.exists(driver_path):
            os.makedirs(os.path.join(self.metabase_dir, "plugins"), exist_ok=True)
            self.download_file("Clickhouse driver", self.clickhouse_driver_url(), driver_path)
        if not os.path.exists(self.metabase_script()):
            self.write_metabase_script()

    def mb_is_running(self):
        proc_iter = psutil.process_iter(attrs=["pid", "name", "cmdline"])
        for p in proc_iter:
            if "metabase.jar" in str(p.info["cmdline"]):
                return True
        return False

    def mb_is_installed(self):
        if os.path.exists(self.metabase_script()):
            return True
            
    def launch_metabase(self):  
        p = subprocess.Popen(["open", '-a', 'Terminal', self.metabase_script()], start_new_session=True)
        self.mb_setup_token = None
        for x in range(60):
            time.sleep(1)
            try:
                r = requests.get("http://localhost:3000/api/session/properties", timeout=0.2)
                if r.status_code == 200:
                    self.mb_setup_token = r.json()["setup-token"]
                    return self.mb_setup_token
            except requests.exceptions.RequestException as e:
                pass
        if not self.mb_setup_token:
            raise RuntimeError("Metabase didn't start successfullly")

    def setup_metabase(self, ch_host:str, ch_database: str, ch_user: str, ch_password: str,
                       mb_password: str, mb_email: str):
        data = {
            "token": self.mb_setup_token,
            "user": {
                "password_confirm": mb_password,
                "password": mb_password,
                "site_name": "Unify personal warehouse",
                "email": mb_email,
            },
            "database": {
                "engine": "clickhouse",
                "name": "unfiydb",
                "details": {
                    "dbname": ch_database,
                    "host": ch_host,
                    "port": 8123,
                    "user": ch_user,
                    "password": ch_password,
                    "ssl": False,
                    "tunnel-enabled": False,
                    "advanced-options": False
                    },
                "is_full_sync": True,
                "database": {
                    "dbname": ch_database,
                    "host": ch_host,
                    "port": 8123,
                    "user": ch_user,
                    "password": ch_password,
                    "ssl": False,
                    "tunnel-enabled": False,
                    "advanced-options": False
                }
            },
            "invite": None,
            "prefs": {
                "site_name": "Unify personal warehouse",
                "site_locale": "en",
                "allow_tracking": True
            }
        }
        r = requests.post("http://localhost:3000/api/setup", json=data)
        if r.status_code != 200:
            raise RuntimeWarning("Metabase setup failed: " + r.text)
        else:
            print("Metabase setup succeeded")

    def open_metabase(self):
        webbrowser.open("http://localhost:3000")
